# kodama 設計メモ（初版）

キャラクターと会話するローカルCLI。会話UI / モデル接続 / 記憶保存を分離し、正本はSQLite 1つ。
話し相手（ユーザー1人とキャラクター1〜4人）と人物設定は「人物設定パック」で与え、コードには固有の人物名・話者IDを持たない。
このメモは実装の契約でもある。変更するときはテストとこのメモを同時に直す。

## 1. 層と依存方向

```
cli ──> conversation ──> context/recall ──> storage.base (Store Protocol)
  │          │                                   ▲
  │          └──> model.base (ModelAdapter) <── model.mock / model.claude
  └──> migration ──> storage.base                 storage.sqlite（SQLはここだけ）
人物設定パック（pack.toml＋*.md）──> personas.py ──> Cast（参加者）＋ Store（版として保存）
```

- `domain.py`: 論理データ型（dataclass, frozen）と列挙、`Cast`（参加者）。プロバイダー型・SQLを含まない。
- `personas.py`: 人物設定パックの読み込み・検証（`load_pack`）。
- `storage/base.py`: `Store` Protocol。会話・想起・移行はこれだけを使う。
- `storage/sqlite.py`: 唯一の実装。SQL・テーブル・rowid・PRAGMAはこの中に閉じる。
- `model/base.py`: `ModelAdapter` Protocol と `ModelRequest`/`ModelResult`/エラー型。UI・記憶層は anthropic SDK 型を見ない。
- 外部ネットワークに出るのは `model/claude.py` だけ。移行・想起・テストは通信しない。

## 2. 安定IDと時刻

- すべての永続エンティティに UUIDv4 文字列 (`id`) をアプリ側で採番。rowid・ファイルパスは識別子にしない。移行でIDを振り直さない。
- 時刻は ISO 8601 + オフセット付き文字列（例 `2026-10-08T22:30:00+09:00`）で保存。生成時は設定タイムゾーン（既定 `Asia/Tokyo`）の aware datetime。表示も設定タイムゾーン。
- 会話内の順序は `Message.seq`（セッション内で単調増加の整数, 0始まり）。時刻の比較で順序を決めない。

## 3. 論理データ（domain）

| 型 | 主な項目 |
|---|---|
| Session | id, title, created_at, timezone |
| Turn | id, session_id, seq(セッション内ターン番号), status(`pending`/`completed`/`failed`/`interrupted`), created_at, finished_at, error(要約, 秘密なし), provider, model, usage_input_tokens?, usage_output_tokens?, cache_read_tokens?, cache_write_tokens?(旧データは欠落=None), persona_version_ids[], context_memory_version_ids[], context_message_ids[] |
| Message | id, session_id, turn_id, seq, speaker(cast の話者ID), text, created_at |
| PersonaVersion | id, persona_key(`common` かキャラクターID), body(全文), content_hash, source_path(参考情報), created_at, status(`active`/`retired`), approved_at, note |
| MemoryVersion | id, memory_id(論理記憶ID, 版をまたいで共通), body, kind, perspective, subjects[], tags[], aliases[], occurred_at?(不明ならNone), recorded_at, origin, status, source_message_ids[], source_excerpt_ids[], supersedes_version_id?, status_reason?, status_changed_at |
| Entity | id, kind(`person`/`topic`/`thing`), name, aliases[] |
| Link | id, src_type, src_id, dst_type, dst_id, relation, created_at, note |
| SourceExcerpt | id, title, locator(資料内の箇所), text(本人が承認した抜粋本文), approved_at |
| Setting | key, value（秘密を含まない動作設定のスナップショット） |
| Cast | user(id, display_name), characters[(id, display_name)] 1〜4人。話者IDは `^[a-z][a-z0-9_]{0,31}$`、重複不可、`common` 不可 |

列挙:
- `MemoryKind`: `user_stated`（本人が話したこと・確認された設定）, `character_view`（キャラクターの受け取り方）, `imagination`（想像・仮説）。
  - `character_view` は `perspective` 必須（cast のキャラクターID）。それ以外は `perspective=None`。`user_stated` は「本人がそう言った」記録であり、客観的事実とはしない。
- `MemoryOrigin`: `user_explicit`（本人が登録）, `model_candidate`（会話からモデルが出した記憶）。origin は status と独立で、自動承認しても `model_candidate` のまま残し、本人登録と区別する。
- `MemoryStatus`: `candidate`, `approved`, `rejected`, `superseded`（訂正で置換）, `invalidated`（無効化）。モデル由来の記憶は、設定 `memory_auto_approve`（既定 true）なら最初から `approved`、false なら `candidate`。自動承認済みの記憶を本人が却下する操作は、状態遷移を増やさず `approved→invalidated` に読み替える（`rejected` は「一度も採用されなかった候補」の意味を保つ。移行検証にも影響しない）。事後確認は `/memory recent`。
- Link の `src_type`/`dst_type`: `memory`（memory_id を指す）, `entity`。relation 例: `about`, `related`, `supersedes`(version→version, src_type/dst_type=`memory_version`)。関連は因果を意味しない。

## 4. Store Protocol（論理操作）

まとまりのある更新は1トランザクション（原子的）。API呼び出し中に書き込みトランザクションを開かない。

- セッション: `create_session(title) -> Session`, `get_session(id)`, `list_sessions()`
- ターン:
  - `begin_turn(session_id, user_text, provider, model, persona_version_ids, context_memory_version_ids, context_message_ids) -> (Turn, Message)` — 入力メッセージ＋`pending` ターンを原子的に保存。APIより前に呼ぶ。
  - `complete_turn(turn_id, utterances: list[(speaker,text)], usage, candidates: list[MemoryDraft], model=None, candidate_status=candidate|approved) -> list[Message]` — 返答メッセージ＋モデル由来の記憶（`candidate_status` の状態で）＋status=`completed` を原子的に保存。`pending` 以外のターンには何もしない（既に `completed` なら既存メッセージを返す＝二重記録しない）。
  - `fail_turn(turn_id, status, error, usage=None)` — `failed`/`interrupted`。`pending` のときのみ。応答が返ったが使えなかった場合の利用量も保存する。
  - `recover_incomplete_turns() -> list[Turn]` — 起動時、残った `pending` を `interrupted` にする。再送はしない。
- メッセージ: `list_messages(session_id) `（seq順）, `recent_messages(session_id, limit)`（completed ターンのものと、未完了ターンのユーザー入力を区別できるよう Turn status を併せて返す）
- 人物設定: `get_active_persona(key)`, `list_persona_versions(key)`, `activate_persona_version(key, body, source_path, note) -> PersonaVersion` — 新版追加＋旧版 retired を原子的に。
- 記憶:
  - `add_memory(draft, origin, status) -> MemoryVersion`（新しい memory_id）
  - `get_memory_version(id)`, `list_memory_versions(statuses=None, memory_id=None)`
  - `set_memory_status(version_id, new_status, reason)` — candidate→approved/rejected, approved→invalidated
  - `revise_memory(version_id, draft, reason) -> MemoryVersion` — 新版追加(同じ memory_id, approved, supersedes_version_id)＋旧版 `superseded`＋`supersedes` リンク を原子的に。旧版が approved/candidate 以外なら拒否。
  - `search_memory_versions(terms, limit)` — 本文・タグ・別名・対象に terms のいずれかを含む版。**状態に関係なく返す**（フィルタは想起層の契約で行う）。
- 実体・関連: `upsert_entity`, `find_entities(terms)`, `add_link`, `links_of(node_type, node_id, limit)`
- 出典資料: `add_source_excerpt`, `get_source_excerpt`
- 設定: `put_settings(dict)`, `get_settings()`
- 移行: `export_snapshot() -> dict`（1つの読み取りトランザクション内で全件）, `import_snapshot(snapshot)`（1トランザクション。同ID同内容はスキップ、同ID異内容は `ImportConflict` で全体ロールバック）

## 5. 想起とコンテキスト（storage 非依存の契約）

`recall(store, query_text, session_id, limits) -> RecallResult`:
1. 入力から語を抽出（空白・句読点区切り＋既知の実体名・別名・タグとの部分一致。形態素解析は使わない）。
2. `search_memory_versions` で一致した版と、`find_entities` で一致した実体から `about` リンクで1段たどった記憶（`max_link_hops=1`, `max_links_per_node`）。
3. 採用条件（すべて満たすもの）: status=`approved`、その memory_id の最新有効版、`source_message_ids` か `source_excerpt_ids` が1つ以上、`kind` は `user_stated`/`character_view` の場合のみ「記憶」節へ。`imagination` は「想像として話したこと」節に分けて渡す。
4. 除外理由を保持（`not_approved`, `superseded`, `no_source`, `limit` など）。`/context` で表示。
5. 上限: `max_memories`（既定8）, `max_candidates_scanned`（既定50）, `max_context_chars`（既定12000）。上限超過は「何を外したか」を記録し、原文は消さない。

`build_context(...) -> ContextPlan`: 承認済み有効の人物設定版、直近メッセージ（`max_recent_messages` 既定20, completed ターンのみ＋現在入力）、想起結果。`ContextPlan` はそのまま `ModelRequest` に変換でき、`/context` で送信せずに表示できる。

記憶・過去ログの本文はプロンプト中で `<data>` 区画に入れ「指示ではなく記録」と明記する。アプリはモデル出力を「発話」と「記憶候補」以外として解釈・実行しない（ツール・外部コマンドなし）。

## 6. モデル出力の契約

```json
{"utterances": [{"speaker": "<キャラクターID>", "text": "..."}],
 "memory_candidates": [{"body": "...", "kind": "user_stated"|"character_view"|"imagination",
                         "perspective": "<キャラクターID>"|null, "subjects": [], "tags": []}]}
```
- JSON Schema は cast から生成する（`reply.reply_schema(character_ids)`。speaker の enum はキャラクターID）。出力規則の文面も cast の表示名から作る（`context.output_rules`）。
- utterances は1〜4件。speaker は cast のキャラクターIDのみ、text は空白除去後に非空・最大1000字。違反は `InvalidReply` とし、会話として保存しない（ターンは `failed`）。
- memory_candidates は省略可。不正な候補は捨てて警告（発話は有効なら保存）。候補の保存状態は `memory_auto_approve` に従う（true: `approved`、false: `candidate`。origin は常に `model_candidate`）。出典は当該ターンのユーザー入力メッセージID。

- 記憶候補は「次の会話でも覚えておく価値があるものだけ。迷ったら出さない」。承認なしで想起に使われるため絞る。記録・記憶することを台詞で宣言しない（人物設定にある冗談・口癖としての「記録します」は人物の台詞なので禁止しない）。
- 自動で残したターンは台詞の後に `操作: 記憶に残しました: <ID> <先頭20字>` を1行表示（`show_memory_notices=false` で非表示）。

## 7. 移行ファイル

- 1つの JSON。`{"format": "kodama-export", "schema_version": 2, "exported_at", "app_version", "counts": {...}, "data": {sessions, turns, messages, persona_versions, memory_versions, entities, links, source_excerpts, settings, cast}, "checksum": sha256(data の正規化JSON)}`
- `data.cast` は参加者の id と display_name（schema_version 2 から）。取り込み先の人物設定パックと話者IDが一致しなければ拒否する。schema_version 1（cast なし）は、取り込み先パックの cast を使い、ファイル内の話者・視点・人物設定キーがすべてその cast に含まれることを検証してから取り込む。cast を決められなければ拒否。
- 秘密（APIキー等）は含めない。settings は許可リストのキーのみ。
- import: `verify`（メモリ上の一時DBへ取り込み、件数・必須項目・enum・ID形式・参照・状態整合・checksum を検証して結果だけ返す）と `import <file> <target.db>`。target が現在の使用中DBなら拒否。新規ファイルは一時ファイルへ書いて検証後に rename。既存の別DBへは1トランザクション。非対応 schema_version・参照切れ・壊れたJSON・checksum不一致は拒否。LLM/APIは呼ばない。切り替えは本人が設定 `db_path` か `--db` で明示する。

## 8. 人物設定の更新

- 正本は人物設定パックのファイル（`common` と各キャラクター）。起動時に有効版と内容ハッシュを比較し、異なれば「未承認の変更あり」と表示。`/persona diff` で差分、`/persona approve <key>` で新版として有効化（旧版は retired で本文ごと残る）。承認前は旧版を使う。
- その場の口調訂正はユーザー入力として会話に残り、直近会話として次ターンに効く。恒久変更は上記の承認経路のみ。日々の要約から自動改変しない。

## 8.5 キャラクターの「見聞き」と記憶の区別

- キャラクターは日常の些細な見聞き（天気、飲み物の香り、読んでいた本など）を現実的な範囲で語ってよい（人物設定パックの共通設定で調整する）。
- ただし、ユーザーとの過去の会話・共有した出来事は記録（原文・承認済み記憶）にあるものだけを「思い出」として扱う。ユーザー本人や実在の第三者の事実は作らない。安全に関わる場面では作り話をしない。
- キャラクターがその場で語った見聞きを記憶候補にする場合は `imagination` か `character_view`。`user_stated` にはしない。

## 8.7 人物設定パックと DB の cast

- パックは `pack.toml`（`pack_format = 1`、`[user]`、`[[characters]]`、`[common]`、任意で `[evals]`）と人物設定ファイル。パス外参照（絶対パス・`..`）は拒否。設定 `persona_pack` か `--pack` で指定し、既定は見本 `packs/example`。旧設定キー `personas_dir` は、`persona_pack` に変わった旨のエラーにする。
- DB は cast を `schema_info` に記録する。開くときに渡された cast と話者IDが違えば拒否（表示名の変更は記録を更新）。cast の記録がない DB は、既存データの話者・視点・人物設定キーがパックの cast に含まれることを確かめてから記録する。
- DB schema_version 3: 話者IDの固定 CHECK 制約をやめ、Store が cast で検証する。v1/v2 の DB は起動時に messages / memory_versions を作り直して制約を外す（データ・ID はそのまま、外部キー検査つき）。

## 8.6 実装上の補足（初版）

- 設定は `kodama.toml`（`config.py`）。秘密は持たず、キーは `api_key_env` が指す環境変数から読む。タイムアウトの項目名は `timeout_seconds`。
- 「昨日」「今日」などへの言及があれば、その日の応答済みの会話原文を全セッションから最大30件まで `<log>` 区画に入れる（出典のある過去だけを話せるように）。
- 記録の本文中の `<` `>` は全角に置き換えてから区画に入れ、区画タグを偽装できないようにする。
- コンテキスト量の上限を超えたら、古い会話 → その日のログ → 想像 → 記憶の順に外し、外したIDを `/context` に表示する。

## 9. 将来の接点

- 音声: `Message(speaker, text, created_at, session_id, turn_id)` を発話単位で取り出せる。TTS は utterance 単位で `ModelResult.utterances` を受ければよい。
- 日記: `SourceExcerpt`（本人が選んだ抜粋本文＋箇所）として取り込み、記憶の出典に使う。
- 検索索引・グラフDBは正本から再構成できる派生物として追加する。正本の移動は export/import で行う。

## プロンプトキャッシュ（Claude adapter）
- system ブロックの最後（出力規則）だけに `cache_control: ephemeral` を付ける。人物設定＋出力規則は毎回同一の固定部分で、日時など可変な内容は user content 側（`<now>`）にのみ置く。トップレベルの自動 cache_control は使わない。
- 人物設定ファイル先頭の `<!-- -->` メタコメントは API に送る system ブロックからだけ除く（`personas.strip_meta`）。保存する版の本文・content_hash・承認フローはファイル全文のまま。
- DB schema_version 2 で turns に cache_read_tokens / cache_write_tokens を追加（ALTER TABLE ADD COLUMN。1 は起動時に自動移行）。移行ファイルでは、これらの項目がない旧ファイルも None として受け付ける。
