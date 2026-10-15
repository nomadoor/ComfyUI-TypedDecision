# ComfyUI-TypedDecision

[English](README.md) · 日本語 · [中文](README_zh.md)

マルチモーダル LLM に文章を書かせず、判定だけをさせるカスタムノードです（Jev 方式の typed decision）。画像・動画・テキストについて型の決まった質問をすると、答えと較正された確率を返します。

- **noul**: はい / いいえ
- **choice**: 選択肢から 1 つ
- **score**: 順序のある段階のどこか

推論は ComfyUI 本体の LLM 推論エンジン（モデル実装・ローダー・メモリ管理）をそのまま使うので、ほぼネイティブに動きます。追加の依存パッケージはありません。

## 対応モデル

| モデル | 公開日 | ベース | int8 のサイズ | 画像 | 動画 |
| --- | --- | --- | --- | --- | --- |
| [imajev-4b](https://huggingface.co/mohit67890/imajev-4b) | 2026-09-24 | Qwen3.5-4B + LoRA | 5.76 GB + 490 MB | 2 枚まで | × |
| [Clef-flash](https://huggingface.co/Cloudflare/clef-flash) | 2026-09-30 | Qwen3.5-9B | 12.2 GB | 何枚でも | ○ |
| [d3-flash](https://huggingface.co/vllm-sr/d3-flash) | 2026-10-10 | Qwen3.5-9B | 9.88 GB | 何枚でも | ○ |
| [d3-mini](https://huggingface.co/vllm-sr/d3-mini) | 2026-10-10 | Qwen3.5-4B | 5.52 GB | 何枚でも | ○ |
| [d3-nano](https://huggingface.co/vllm-sr/d3-nano) | 2026-10-10 | Qwen3.5-2B | 3.06 GB | 何枚でも | ○ |
| [d3-lite](https://huggingface.co/vllm-sr/d3-lite) | 2026-10-10 | Qwen3.5-0.8B | 1.21 GB | 何枚でも | ○ |

詳しくはブログへ: [日本語](https://comfyui.nomadoor.net/ja/notes/typed-decision/) · [English](https://comfyui.nomadoor.net/en/notes/typed-decision/) · [中文](https://comfyui.nomadoor.net/zh/notes/typed-decision/)

## インストール

ComfyUI Manager で **ComfyUI-TypedDecision** を検索してインストールしてください。

## モデル

### imajev

Qwen3.5-4B と imajev-4b のフォルダを使います。

| | int8 ConvRot | bf16 |
| --- | --- | --- |
| Qwen3.5-4B | [qwen3.5_4b_int8_convrot](https://huggingface.co/nomadoor/Qwen3.5/blob/main/text_encoders/qwen3.5_4b_int8_convrot.safetensors) (5.76 GB) | [qwen3.5_4b_bf16](https://huggingface.co/Comfy-Org/Qwen3.5/blob/main/text_encoders/qwen3.5_4b_bf16.safetensors) (9.32 GB) |

[mohit67890/imajev-4b](https://huggingface.co/mohit67890/imajev-4b/tree/main) の下記ファイル (490 MB) を `imajev-4b` フォルダに入れます。リポジトリごとダウンロードしても動きます。

```text
📂ComfyUI/
└── 📂models/
    ├── 📂text_encoders/
    │   └── qwen3.5_4b_int8_convrot.safetensors
    └── 📂typed_decision/
        └── 📂imajev-4b/
            ├── adapter_config.json
            ├── adapter_model.safetensors
            ├── calibration-rot4-modality.json
            ├── decision_readout.json
            └── decision_readout.safetensors
```

### d3 / Clef-flash

ファイル 1 つにモデル全体（判定ヘッド込み）が入っています。使うものを `text_encoders` に置きます。

| | int8 ConvRot | bf16 |
| --- | --- | --- |
| d3-flash | [d3_flash_int8_convrot](https://huggingface.co/nomadoor/d3/blob/main/text_encoders/d3_flash_int8_convrot.safetensors) (9.88 GB) | [d3_flash_bf16](https://huggingface.co/nomadoor/d3/blob/main/text_encoders/d3_flash_bf16.safetensors) (16.8 GB) |
| d3-mini | [d3_mini_int8_convrot](https://huggingface.co/nomadoor/d3/blob/main/text_encoders/d3_mini_int8_convrot.safetensors) (5.52 GB) | [d3_mini_bf16](https://huggingface.co/nomadoor/d3/blob/main/text_encoders/d3_mini_bf16.safetensors) (9.08 GB) |
| d3-nano | [d3_nano_int8_convrot](https://huggingface.co/nomadoor/d3/blob/main/text_encoders/d3_nano_int8_convrot.safetensors) (3.06 GB) | [d3_nano_bf16](https://huggingface.co/nomadoor/d3/blob/main/text_encoders/d3_nano_bf16.safetensors) (4.43 GB) |
| d3-lite | [d3_lite_int8_convrot](https://huggingface.co/nomadoor/d3/blob/main/text_encoders/d3_lite_int8_convrot.safetensors) (1.21 GB) | [d3_lite_bf16](https://huggingface.co/nomadoor/d3/blob/main/text_encoders/d3_lite_bf16.safetensors) (1.71 GB) |
| Clef-flash | [clef_flash_int8_convrot](https://huggingface.co/nomadoor/clef/blob/main/text_encoders/clef_flash_int8_convrot.safetensors) (12.2 GB) | [clef_flash_bf16](https://huggingface.co/nomadoor/clef/blob/main/text_encoders/clef_flash_bf16.safetensors) (19.1 GB) |

```text
📂ComfyUI/
└── 📂models/
    └── 📂text_encoders/
        ├── d3_mini_int8_convrot.safetensors
        └── clef_flash_int8_convrot.safetensors
```

## ノード

### Load Typed Decision Model

`backbone` でモデルを選び、`adapter` で対応するものを選びます。

| モデル | `backbone` | `adapter` |
| --- | --- | --- |
| imajev | Qwen3.5 | `imajev-4b` |
| d3 | d3 のファイル | `d3` |
| Clef-flash | Clef-flash のファイル | `clef` |

`adapter` を選ぶと、そのモデルの設定が出ます。

- `debias`（imajev, d3）: 選択肢の並び順を変えて判定し、平均します。遅くなります。
  - imajev: すべての mode で、最大 4 通りの並び順
  - d3: choice のみ、逆順も判定
- `fps`（d3, Clef-flash）: 動画を 1 秒に何フレーム読むか（既定 2）
- `calibration`（imajev）: 温度較正をかける

### Typed Decision

質問を 1 つ判定します。質問と選択肢は英語で書きます。

| 入力 | 内容 |
| --- | --- |
| `mode` | `noul` / `choice` / `score` |
| `instructions` | 質問 |
| `criteria` | choice の選択肢、score の段階（1 行に 1 つ） |
| `criteria_true` / `criteria_false` | noul ではい / いいえとする条件（任意） |
| `images` | 判定する画像。imajev で 2 枚なら 1 枚目が参照、2 枚目が判定対象 |
| `videos` | 判定する動画（d3, Clef-flash） |
| `state` | 判定の前提にする事実。テキストか JSON（任意） |
| `threshold` | `pass` の閾値 |

| 出力 | 内容 |
| --- | --- |
| `value` | noul: はいの確率<br>choice: 選ばれた選択肢の確率<br>score: 段階の期待値（1 行目 = 0） |
| `label` | 答え（`threshold` に影響されない） |
| `index` | noul: はい = 1<br>choice, score: 答えの行番号（0 から） |
| `pass` | `value` >= `threshold` かつ棄権していないとき true |
| `abstained` | 画像と state では答えられないとき true（imajev のみ） |
| `result` | 答えの全体（JSON） |

#### 動画について

- 動画全体を loader の `fps` で読み、全フレームを 1 回で判定します。1 分を 2 fps なら 120 フレームです。
- d3 は長い動画で急に遅くなります（d3-mini int8・12 GB で 20 秒が 4 秒、70 秒が 5 分）。長い動画は `fps` を下げてください。
- Clef-flash は動画 1 本で画素数の上限を分け合うので、長い動画ほど小さく読みます。
- 読んだフレーム数と間隔は `result` に入ります。

### Typed Decision Moment Search

動画の中で条件が成り立つ区間を探します（例: 「A dog is visible.」）。1 フレームで分かる条件向けで、動きの向きなどは判定できません。

| 入力 | 内容 |
| --- | --- |
| `instructions`, `criteria_true`, `criteria_false`, `state` | Typed Decision の noul と同じ |
| `images` | 参照画像（任意）。画像の人や物を動画から探すとき |
| `video` | 探す動画 |
| `window` | 最初にまとめて判定するフレーム数（既定 8 = 4 秒分） |
| `threshold` | フレームを合格にする確率 |

| 出力 | 内容 |
| --- | --- |
| `intervals` | 区間の JSON: `start`, `end`（秒）、`start_frame`, `end_frame`（元動画のフレーム番号、end は含まない）、`score`（区間内の最高確率） |
| `videos` | 区間ごとに切り出した動画。見つからなければ後ろの動画ノードは実行されない |

#### 仕組み

動画を 2 fps で読み、3 段階で判定します。

1. `window` フレームずつまとめて判定し、外れたブロックを丸ごと飛ばす
2. 残ったブロックを 1 フレームずつ判定する
3. 合格と不合格のフレームの間を元動画のフレーム単位で調べ、境界を決める

#### 注意

- 参照画像を使うときは動画について質問します。「The video shows the same person as the image.」は使えますが、「The person in the image is visible.」は参照画像自体に当てはまり、全フレームが合格になります。
- `window` を大きくすると速くなる代わりに、短い出来事を落としやすくなります（16 では d3-mini と d3-flash が 0.5 秒の出来事を落とすことがある）。imajev は 2 まで、参照画像を使うなら 1 です。
- 区間はモデルの判定どおりです。画面から半分出た物はもう「いない」とされることがあるので、数えたいものに合わせて質問を書いてください。
- 0.5 秒より短い出来事は、2 fps のフレームの間に落ちることがあります。

## サンプルワークフロー

- [noul](example_workflows/typed_decision_noul.json): 画像に人がたくさん写っているか
- [choice](example_workflows/typed_decision_choice.json): 主な被写体は何か
- [score](example_workflows/typed_decision_score.json): 学習用の写真としてどれくらい使えるか
- [2 枚の画像](example_workflows/typed_decision_noul_two_images.json): 2 枚に同じ人が写っているか
- [データセット仕分け](example_workflows/typed_decision_noul_dataset_sort.json): 人の有無で写真を別の名前で保存する
- [プロンプトから解像度](example_workflows/typed_decision_choice_resolution_from_prompt_krea2.json): プロンプトからアスペクト比を選んで Krea 2 で生成する
- [Moment Search](example_workflows/typed_decision_moment_search_d3_mini.json): 参照画像の人が映っている区間を d3-mini で探し、つなげて保存する

## クレジット

- [imajev](https://github.com/mohit67890/imajev) と [imajev-4b](https://huggingface.co/mohit67890/imajev-4b)（mohit67890）: モデル、プロンプト形式、スコアリング
- [d3 (Decision 3.0)](https://huggingface.co/vllm-sr/d3-flash)（vLLM Semantic Router チーム）: モデル、プロンプト形式、readout
- [Clef-flash](https://huggingface.co/Cloudflare/clef-flash)（Cloudflare）: モデル、プロンプト形式、joint schema head
- [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-4B)（Qwen チーム）。ComfyUI 向けの再パッケージは [Comfy-Org](https://huggingface.co/Comfy-Org/Qwen3.5)
