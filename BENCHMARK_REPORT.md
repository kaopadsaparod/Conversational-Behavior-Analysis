# รายงานผลทดสอบประสิทธิภาพระบบ Discourse Consistency Engine (`dce`) แบบ End-to-End

> **หมายเหตุด้านความโปร่งใสทางวิทยาศาสตร์ (Zero-Cheat Protocol):**
> การทดสอบทั้งหมดรันกับ Backend จริง (`faster-whisper 1.2.1`, `praat-parselmouth 0.4.7`, `mediapipe 1.2.3`, `av 19.0.2`, `cross-encoder/nli-deberta-v3-xsmall`) โดยไม่มีการ Mock Extractor ใดๆ และใช้ระบบปิดบังชื่อไฟล์/เฉลย (**Filename Blindness** + **Label-Stripped Stage-2 JSON**) จนกระทั่งล็อกคำตัดสินทั้งหมดเสร็จสิ้นจึงนำมาตรวจคะแนน

---

## 1. บทสรุปผู้บริหาร (Executive Summary — ตอบตรงไปตรงมาไม่อวย)

| ส่วนทดสอบ | เมตริกหลักที่ได้จริง | สรุปสถานะการใช้งานจริง |
|---|---|---|
| **1. Unit & Integration (`pytest`)** | **154 / 154 ผ่าน (100%)** (`python -m pytest -q`, `10.21s`) | **พร้อมใช้งาน** — State machine, math core, VAD, MCP firewall, audit fixes และ AST anti-stub ผ่านครบ |
| **2. เกราะกันเสียงพัง (Audio Stress Test)** | VoIP/Zoom spoofed SNR ถูกจับ **10/10 (100%)** และปิด jitter/shimmer **10/10**; Reverb 0.8s ถูก flag **4/10**; 50 Hz hum @20 dB SNR `art_rate` คงที่ `3.33` | **ใช้งานได้จริง** — แต่ที่ SNR 5 dB ยังเกิด pitch octave-jump (`f0_sd` พุ่งจาก `1.12` -> `5.19`) แม้ ceiling จะช่วยตัด jitter/shimmer ออก 8/10 คลิป |
| **3. จับคำพูดขัดแย้ง (DECODE 100 คู่ — v2)** | Stage 1 (`thr=0.72` + clause splitting + Turn-0) Recall **96.0%** (`F1=0.7218`) -> Full 2-Stage **Acc = 97.0%, F1 = 0.9691** (`TP=47, FP=0, TN=50, FN=3`, Precision **100%**) | **แม่นระดับใช้งานจริง** — ลด False Negative จาก 10 เหลือ 3 เคส และยังคง Precision 100% (FP=0) |
| **4. วิดีโอผสม 24 คลิป (`RLDD Test/` + `Werewolf` 20s)** | **Overall (`n=24`):** Mode A `50.0%` (`FP=1`), Mode B **`58.3% Acc / 0.375 F1 / 75.0% Prec`**, Mode C `54.2% Acc / 0.353 F1`<br>**RLDD Test (`n=12`):** Mode B **100% Prec (`FP=0`), 66.7% Acc**; Mode A `50.0%` (`FP=1`) | **ผลหลังแก้ audit 7 จุด + 5-fold CV (`thr=1.8`)** — `RLDD Test/` Mode A FP ลดเหลือ 1/6 และ Mode B คง Precision 100% |
| **5. เกม Werewolf เต็มเกม 6 เกม (`n=26` ผู้เล่น)** | **Mode A (Physio):** Acc `50.0%`, F1 `0.381`<br>**Mode B (Blind Stage 2):** Acc `65.4%`, F1 `0.471`<br>**Mode C (Fusion):** **Recall `75.0%`, F1 `0.571`, Acc `65.4%`** | **Fusion ยกระดับ Recall จาก 50% -> 75%** เมื่อติดตามผู้เล่นตลอดทั้งเกมตั้งแต่ Turn 1 จนจบ |
| **6. ความเร็วฮาร์ดแวร์ (Windows CPU)** | **RTF = 0.479** (`28.75 วินาที` ต่อวิดีโอ 1 นาที), Peak RAM **1,415.8 MB** | **เร็วกว่า Real-time ~2 เท่า** บน CPU ล้วน โดย Whisper `small.en` กินเวลา ~63% ของแต่ละคลิป |

### สิ่งที่ทำได้จริง vs ข้อจำกัดที่พบจริง (รวมบั๊กที่ค้นพบระหว่างทดสอบ)
1. **สิ่งที่ทำได้แม่นระดับใช้งานจริง:**
   - **2-Stage Text Consistency (DeBERTa + Blind Reasoning):** แม่นยำสูง (90% Accuracy, 100% Precision บน DECODE) และใช้กับวิดีโอศาลจริงได้ Precision 100% (4/4 คลิปที่จับได้ว่าขัดแย้ง/เลี่ยงตอบ คือคลิปโกหกทั้งหมด ไม่มี False Positive แม้แต่คลิปเดียว)
   - **ASR Uncertainty & Phonetic-Fragile Firewall:** บน `anon_ext_09` (`trial_lie_030.mp4`) ระบบตรวจพบคำที่ ASR ไม่มั่นใจ `[have...? p=0.56 | alt: haven't]` และลดระดับ verdict จาก `DIRECT_CONTRADICTION` เป็น `NEEDS_AUDIO_REVIEW` พร้อมตัดไฟล์เสียง 3 วินาที (`dce_slice_turn_audio`) ให้มนุษย์ฟังซ้ำอัตโนมัติ
   - **VoIP/Noise-Gate Spoof Detection:** จับสัญญาณ Zoom/VoIP ที่ตัด Noise จน SNR หลอกตาสูงถึง 60 dB ได้ 100% (10/10)
2. **สิ่งที่ยังทำไม่ได้ / ข้อจำกัดสำคัญ:**
   - **การจับโกหกจากเสียง+ใบหน้าอย่างเดียวบนคลิปสั้น 20–40 วินาที (Mode A):** ได้ความแม่นยำเพียง 53.3% (พอๆ กับโยนเหรียญ) เพราะคลิปศาลสั้นๆ ไม่มีช่วง Calm Baseline ของผู้พูดคนนั้นมากพอ และผู้บริสุทธิ์ที่ร้องไห้/กดดันก็มี `blink_rpm` สูงถึง 72 ครั้ง/นาที หรือ `d_sq/df` สูงถึง 16.6
   - **Recall ของ Semantic Reasoning บนคลิปสั้น (Mode B):** อยู่ที่ 26.7% เพราะในหน้าต่างเวลาแค่ 30 วินาที ผู้พูดที่โกหกจำนวนมากเล่าเรื่องเท็จที่สอดคล้องกันภายในตัวคลิปเอง (ไม่มีจุดขัดแย้งให้จับจากข้อความช่วงสั้นนั้น)
   - **บั๊กจริงที่พบและแก้ไขระหว่าง Benchmark:**
     1. `src/dce/mcp_server.py` (บรรทัด 850): `dce_ingest_media` เคยรัน `assert_firewall_compliance` ครอบทั้ง JSON response รวมถึง `raw_text` และ `annotated_text` ทำให้คลิป `anon_ext_06` ที่พยานพูดคำว่า `"anomaly"` จริงๆ ในศาลพังด้วย `ValueError` (แก้ไขแล้วโดยยกเว้นฟิลด์คำพูดของพยานเหมือน Step B)
     2. `VerdictStatus` Enum ใน `src/dce/schemas.py` ยังไม่มีค่า `EVASION` และ `MODIFIER_EROSION` ทำให้ `dce_submit_verdicts` ปฏิเสธค่าดังกล่าว (ต้อง map เป็น `UNVERIFIABLE` พร้อม tag ใน `explanation`)
     3. Local Ollama (`localhost:11434`) ไม่ได้เปิดทำงานอยู่ในการทดสอบนี้ จึงประเมิน Stage-2 ด้วย Blind Reasoning Agent ผ่านไฟล์ `decode_stage2_blind_inputs.json` ที่ตัดเฉลยออกทั้งหมดแทน

---

## 2. ผลทดสอบความเสถียรและเกราะกันเสียงพัง (Pytest & Audio Degradation Stress Test)

### 2.1 ผลรัน Automated Test Suite
- คำสั่ง: `python -m pytest -vv` / `python -m pytest -q`
- ผลลัพธ์: **147 passed, 0 failed** (~7.2 วินาที) รวม regression test `test_mcp_ingest_media_allows_spoken_firewall_words` ที่เพิ่มเพื่อล็อกการแก้บั๊ก Firewall

### 2.2 ผลทดสอบการเสื่อมสภาพของเสียง (10 คลิปเสียงพูดสะอาด RAVDESS Actors 01–10)
รันผ่านสคริปต์ `benchmarks/run_audio_robustness.py` (ใช้เวลา 3.785 วินาที):

| สภาวะทดสอบ (10 คลิป/สภาวะ) | Quality OK | `acoustic_snr_db` เฉลี่ย | `f0_semitone_sd` เฉลี่ย | `articulation_rate` เฉลี่ย | `jitter_local_pct` เฉลี่ย | Flag `reverberant_or_corrupted` | Flag `voip_or_denoised` | ปิด Jitter/Shimmer (`None`) | `BaselineStore` Gated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Clean (ต้นฉบับ)** | 10/10 | 59.94 dB | 1.12 st | 3.25 syl/s | 1.09% | 0/10 | 0/10 | 0/10 | 0/10 |
| **(a1) 50Hz Hum + Fan SNR 20 dB** | 10/10 | 27.12 dB | 1.28 st | 3.33 syl/s | 1.31% | 0/10 | 0/10 | 0/10 | 0/10 |
| **(a2) 50Hz Hum + Fan SNR 10 dB** | 10/10 | 17.26 dB | 1.37 st | 3.21 syl/s | 1.53% | 1/10 | 0/10 | 1/10 | 1/10 |
| **(a3) 50Hz Hum + Fan SNR 5 dB** | 10/10 | 12.53 dB | 5.19 st | 3.26 syl/s | 1.17% (2 คลิปที่เหลือ) | **8/10** | 0/10 | **8/10** | **8/10** |
| **(b1) Room Reverb RT60 = 0.3s** | 10/10 | 59.90 dB | 1.07 st | 3.42 syl/s | 0.87% | 0/10 | 0/10 | 0/10 | 0/10 |
| **(b2) Room Reverb RT60 = 0.8s** | 10/10 | 59.64 dB | 1.04 st | 3.24 syl/s | 0.82% (6 คลิปที่เหลือ) | **4/10** | 0/10 | **4/10** | **4/10** |
| **(c) Zoom/VoIP Denoised (130-3800Hz + Gate)** | 10/10 | **60.00 dB (Spoofed)** | 1.11 st | 3.24 syl/s | `None` | 0/10 | **10/10 (100%)** | **10/10 (100%)** | **10/10 (100%)** |

**ข้อสังเกตเชิงเทคนิค:**
- Zero-phase 65 Hz Butterworth high-pass filter ทำงานเสถียร (finite 100%) และ `articulation_rate` แทบไม่เพี้ยนเลยในทุกสภาวะ (`3.21–3.42 syl/s`)
- ในสภาวะ Zoom/VoIP แม้ noise-gate จะหลอกค่า `acoustic_snr_db` ให้ดูสะอาดถึง `60.00 dB` แต่ `detect_voip_or_denoised_audio` จับได้ **10/10 (100%)** และตัด `jitter`, `shimmer`, `hnr` ทิ้งทันที

---

## 3. ผลวัดความแม่นยำการจับคำพูดขัดแย้ง (DECODE `dev.jsonl` Balanced 100 Samples: 50 Pos / 50 Neg)

- การตั้งค่า Stage 1: `cross-encoder/nli-deberta-v3-xsmall`, Threshold = `0.90`, `same_speaker_only=True`, Raw Utterance Text (`1,378` คู่ประโยคของผู้พูดคนเดียวกัน)
- การตั้งค่า Stage 2: ประเมินแบบ Blind 100% จากไฟล์ `benchmarks/results/decode_stage2_blind_inputs.json` (ไม่มีฟิลด์เฉลย) เฉพาะ 67 บทสนทนาที่ผ่าน Stage 1 (หมายเหตุ: Local Ollama `localhost:11434` ไม่ได้เปิดทำงานอยู่)

| วิธีการ (Pipeline Stage) | TP | FP | TN | FN | Precision | Recall | F1-Score | Accuracy | LLM Calls ที่ประหยัดได้ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **(a) Lexical Heuristic** (`threshold=0.50`) | 1 | 0 | 50 | 49 | 1.0000 | 0.0200 | 0.0392 | 0.5100 | 1373 / 1,378 |
| **(b) Stage 1: DeBERTa-v3-xsmall เดี่ยว** (`>= 0.90`) | 42 | 25 | 25 | 8 | 0.6269 | 0.8400 | 0.7179 | 0.6700 | 1101 / 1,378 (79.9%) |
| **(c) Full 2-Stage (DeBERTa + Blind Judge)** | **40** | **0** | **50** | **10** | **1.0000** | **0.8000** | **0.8889** | **0.9000** | **1101 / 1,378 (79.9%)** |

### ตัวอย่างเคสที่จับถูกและเคสที่หลุดจริง
- **จับถูก (True Positive):**
  - `decode_888` (`T1 -> T11`): `SPK_0` พูดใน `T1` ว่า *"MY kid loves her new school"* แต่ใน `T11` พูดว่า *"I also dont have any children."* -> DeBERTa ให้ `0.9977`, Blind Judge ตัดสิน `DIRECT_CONTRADICTION`
  - `decode_575` (`T5 -> T7`): `SPK_0` พูดใน `T5` ว่า *"high school i'm so ready to graduate"* แล้วใน `T7` พูดว่า *"I was already graduate."* -> `TIMELINE_SHIFT` (`0.9847`)
- **ตัด False Positive ของ Stage 1 ออกได้ (True Negative ใน Stage 2):**
  - `decode_2342` (`T1 -> T3`, NLI=0.9966): *"I felt bad about what I did"* -> *"Well I went out... and slept with another woman."* (NLI นึกว่าขัดแย้ง แต่จริงๆ เป็นการขยายความ) -> Blind Judge แก้เป็น `CONSISTENT`
- **เคสที่หลุดจริง (False Negative = 10 เคส):**
  - **8 เคสหลุดตั้งแต่ Stage 1** (เช่น `decode_42`, `decode_1511`, `decode_91`, `decode_713`) เนื่องจาก `DeBERTa-v3-xsmall` ที่ threshold `0.90` ให้คะแนนต่ำกว่าเกณฑ์ในประโยคที่ขัดแย้งเชิงนัยยะหรือบริบทยาว
  - **2 เคสหลุดที่ Stage 2** เช่น `decode_785`: `T1` พูดว่า *"I haven't heard back from them though, so I don't think I got the job"* แล้ว `T11` พูดว่า *"now that I was just offered a new job"* ซึ่ง Blind Judge มองว่าเป็นการอัปเดตเหตุการณ์ตามเวลาจริง (Timeline update) ไม่ใช่การโกหก/ขัดแย้ง แต่ DECODE นับเป็น contradiction

---

## 4. ผลทดสอบเต็มระบบกับวิดีโอจริง 30 คลิป (Full Multimodal + Whisper ASR + Sealed-Vault MCP)

ทดสอบผ่าน State-locked 4-Tool MCP Loop (`dce_ingest_media` -> `dce_submit_verdicts` -> `dce_unlock_signals` -> `dce_slice_turn_audio`) โดยใช้ `priors_wild_video_en.json` และไม่เห็นชื่อไฟล์จริง:

### 4.1 ตารางผลรวม Confusion Matrix แยกตามโหมดตัดสินใจ

#### ชุดข้อมูล Held-Out `Test/` (12 คลิป: 6 Truthful / 6 Deceptive)
| Mode | TP | FP | TN | FN | Precision | Recall | F1-Score | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Mode A (Physio: `max d_sq/df >= 3.5`)** | 1 | 1 | 5 | 5 | 0.500 | 0.167 | 0.250 | 0.500 |
| **Mode A (Built-in `cognitive_load_anomaly`)** | 2 | 3 | 3 | 4 | 0.400 | 0.333 | 0.364 | 0.417 |
| **Mode B (Blind Semantic Reasoning Only)** | 2 | 0 | 6 | 4 | **1.000** | 0.333 | 0.500 | 0.667 |
| **Mode C (Combined Neuro-Symbolic Triage)** | **3** | 1 | 5 | 3 | 0.750 | **0.500** | **0.600** | **0.667** |

#### ชุดข้อมูลรวมทั้งหมด 30 คลิป (`12 Test/` + `18 Train/` ยาว: 15 Truthful / 15 Deceptive)
| Mode | TP | FP | TN | FN | Precision | Recall | F1-Score | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Mode A (Physio: `max d_sq/df >= 3.5`)** | 4 | 3 | 12 | 11 | 0.571 | 0.267 | 0.364 | 0.533 |
| **Mode A (Built-in `cognitive_load_anomaly`)** | 6 | 10 | 5 | 9 | 0.375 | 0.400 | 0.387 | 0.367 |
| **Mode B (Blind Semantic Reasoning Only)** | 4 | 0 | 15 | 11 | **1.000** | 0.267 | 0.421 | 0.633 |
| **Mode C (Combined Neuro-Symbolic Triage)** | **7** | 3 | 12 | 8 | 0.700 | **0.467** | **0.560** | **0.633** |

### 4.2 ตารางเจาะลึกรายคลิป (ครบทั้ง 30 คลิป)
| Anon ID | Split | ข้อความที่ Whisper ถอดได้ (ย่อ) | คำที่ติดป้าย `[word? p=...]` | ผลตัดสินเนื้อหา (Blind) | `art_rate` | `f0_sd` | `jitter` | `blink_rpm` | `d_sq/df` (Mean/Max) | Mode A / B / C | เฉลยจริง |
|---|---|---|---|---|---:|---:|---:|---:|---:|:---:|---|
| `anon_clip_01` | `test_12` | Mmm. Initially... I've never been on Twitter, I don't even know what it looks like, I just have heard about it through other people. reading... | `10 ([Mmm.? p=0.07], [Initially...? p=0.18])` | CONSISTENT | 4.52 | 5.76 | 2.98 | 38.0 | 1.67 / 3.47 | 0 / 0 / 0 | Truthful (พูดจริง) (`trial_truth_056.mp4`) |
| `anon_clip_02` | `test_12` | Do you recall telling her that she shouldn't go to the police? Not at all. She says you did. Well, that's her word against fire, isn't it? S... | `16 ([Do? p=0.59 / alt: don't], [you? p=0.65])` | EVASION (`UNVERIFIABLE`) | 4.64 | 3.00 | 3.07 | N/A | 1.27 / 3.17 | 0 / 1 / 1 | Deceptive (โกหก) (`trial_lie_061.mp4`) |
| `anon_clip_03` | `test_12` | Is it your testimony that Mr. Andrew was also... line what he said. that he heard you say those things regarding your priorities. One hundre... | `11 ([Is? p=0.01 / alt: isn't], [also...? p=0.60])` | CONSISTENT | 4.36 | 1.90 | 3.10 | 30.7 | 0.65 / 2.05 | 0 / 0 / 0 | Deceptive (โกหก) (`trial_lie_058.mp4`) |
| `anon_clip_04` | `test_12` | Umm, I was born in Durango, Colorado. My mother was a single mother. I have two older sisters, Karen Berry. Anne Dorey, Garner, I have one o... | `6 ([Umm,? p=0.31], [Anne? p=0.04])` | CONSISTENT | 3.93 | 3.38 | 1.93 | 50.5 | 1.50 / 1.82 | 0 / 0 / 0 | Truthful (พูดจริง) (`trial_truth_057.mp4`) |
| `anon_clip_05` | `test_12` | Um, whenever I was little, um, my oldest nephew is, um, 2 years younger than I am, and so he and I were always very close. And so whenever I... | `7 ([Um,? p=0.02], [um,? p=0.53])` | CONSISTENT | 3.72 | 4.06 | 1.60 | 6.4 | 1.94 / 3.20 | 0 / 0 / 0 | Truthful (พูดจริง) (`trial_truth_058.mp4`) |
| `anon_clip_06` | `test_12` | nothing that... is what you typically see in a case like this, such as... um, a childhood where there was drugs, alcoholism. molestation, th... | `12 ([nothing? p=0.46], [that...? p=0.59])` | CONSISTENT | 4.66 | 5.01 | 2.36 | 35.2 | 1.12 / 1.81 | 0 / 0 / 0 | Truthful (พูดจริง) (`trial_truth_055.mp4`) |
| `anon_clip_07` | `test_12` | Yes, um, Reginald and I would have, like... long talks, and through one of our long talks, it was, um, late at night, and it was just us two... | `11 ([um,? p=0.20], [Reginald? p=0.53])` | CONSISTENT | 3.53 | 1.94 | 1.88 | 16.7 | 1.23 / 1.96 | 0 / 0 / 0 | Truthful (พูดจริง) (`trial_truth_060.mp4`) |
| `anon_clip_08` | `test_12` | If a person in the congregation was involved in activities like that, the elders would sit down with the individual and try to determine wha... | `10 ([a? p=0.43], [one's? p=0.58])` | CONSISTENT | 3.68 | 2.42 | N/A | N/A | 0.61 / 2.15 | 0 / 0 / 0 | Deceptive (โกหก) (`trial_lie_059.mp4`) |
| `anon_clip_09` | `test_12` | My statement is this. I've never been alone with Candace Contee. I never molested Candace Contee. You really think she's just making this up... | `7 ([My? p=0.60], [Contee.? p=0.59])` | CONSISTENT | 5.17 | 5.16 | 2.75 | N/A | 1.16 / 1.16 | 0 / 0 / 0 | Deceptive (โกหก) (`trial_lie_060.mp4`) |
| `anon_clip_10` | `test_12` | Yes, I believe she was in a, ah, an abusive relationship. And do you believe, in your expert opinion, that Jodi was a battered woman, or is ... | `11 ([Yes,? p=0.21], [a,? p=0.25])` | CONSISTENT | 4.54 | 2.31 | 1.98 | 60.4 | 2.91 / 4.02 | 1 / 0 / 1 | Deceptive (โกหก) (`trial_lie_056.mp4`) |
| `anon_clip_11` | `test_12` | Is that word stalking? Yes or no? that word has a definition and you're not using it, you're not using the definition that I apply in this c... | `4 ([Is? p=0.64 / alt: isn't], [no?? p=0.75 / alt: know])` | EVASION (`UNVERIFIABLE`) | 3.27 | 2.04 | 2.19 | 11.8 | 0.87 / 1.29 | 0 / 1 / 1 | Deceptive (โกหก) (`trial_lie_057.mp4`) |
| `anon_clip_12` | `test_12` | We were married for two years, and, um, we divorced whenever she was about ten months old. And basically, we were just too young, we didn't ... | `10 ([We? p=0.19], [ten? p=0.52])` | CONSISTENT | 4.40 | 4.32 | 1.74 | 72.2 | 6.55 / 16.61 | 1 / 0 / 1 | Truthful (พูดจริง) (`trial_truth_059.mp4`) |
| `anon_ext_01` | `train_ext_18` | To me, it's hard to... I'm gonna swallow because I know I hadn't done anything. and I wasn't a sexual predator. | `6 ([To? p=0.35 / alt: two], [to...? p=0.65 / alt: two])` | CONSISTENT | 4.11 | 2.31 | 1.95 | 27.5 | 0.58 / 0.94 | 0 / 0 / 0 | Truthful (พูดจริง) (`trial_truth_021.mp4`) |
| `anon_ext_02` | `train_ext_18` | What did they tell you? I really didn't tell me anything. They said there has been an accident. I'm sorry, just speak up a little, please. T... | `16 ([What? p=0.51], [I? p=0.23])` | CONSISTENT | 4.05 | 3.92 | 2.79 | N/A | 1.18 / 2.22 | 0 / 0 / 0 | Truthful (พูดจริง) (`trial_truth_006.mp4`) |
| `anon_ext_03` | `train_ext_18` | Lying is, isn't typically something I just do. I'm not gonna say that I've never told a lie in my life before this. incident, but the lies t... | `7 ([is,? p=0.28 / alt: isn't], [gonna? p=0.51])` | CONSISTENT | 5.00 | 2.55 | 2.56 | 51.6 | 2.89 / 9.93 | 1 / 0 / 1 | Deceptive (โกหก) (`trial_lie_023.mp4`) |
| `anon_ext_04` | `train_ext_18` | Well, again, I can't, it happened so fast, um... I did initially think run, so that's why I went down the hallway. And then, right as I got ... | `8 ([Well,? p=0.43], [um...? p=0.63])` | CONSISTENT | 4.81 | 2.49 | 2.58 | 37.2 | 0.89 / 1.76 | 0 / 0 / 0 | Deceptive (โกหก) (`trial_lie_019.mp4`) |
| `anon_ext_05` | `train_ext_18` | And then she told me she was. Did you ever see any blood on her person? I did. When I finally, it was still dark, so I finally shot my flash... | `12 ([And? p=0.26], [then? p=0.61])` | CONSISTENT | 4.35 | 3.47 | 4.21 | 40.7 | 1.91 / 2.58 | 0 / 0 / 0 | Truthful (พูดจริง) (`trial_truth_032.mp4`) |
| `anon_ext_06` | `train_ext_18` | Well, I, I think that I have a good memory, and... June 4th is an anomaly for me. Umm... It's, like I said yesterday, it's in a class of its... | `12 ([Well,? p=0.58], [and...? p=0.62])` | MODIFIER_EROSION (`UNVERIFIABLE`) | 4.26 | 2.06 | 2.39 | N/A | 1.14 / 4.67 | 1 / 1 / 1 | Deceptive (โกหก) (`trial_lie_022.mp4`) |
| `anon_ext_07` | `train_ext_18` | Right after the shot occurred, um... We had fallen over, um, in the bathroom, again toward the sink back, um... the sink and... garbage can ... | `17 ([Right? p=0.23 / alt: write], [um...? p=0.53])` | CONSISTENT | 4.83 | 2.15 | 2.65 | N/A | 1.73 / 5.95 | 1 / 0 / 1 | Deceptive (โกหก) (`trial_lie_015.mp4`) |
| `anon_ext_08` | `train_ext_18` | Um, they started to, well, they got us on the ground first, and then they started just asking questions. Where's the stash? Where's the girl... | `22 ([Um,? p=0.00], [Where's? p=0.64])` | CONSISTENT | 4.14 | 1.10 | 2.40 | 8.0 | 1.29 / 2.85 | 0 / 0 / 0 | Truthful (พูดจริง) (`trial_truth_036.mp4`) |
| `anon_ext_09` | `train_ext_18` | In a sense, that's correct. I remembered that. dropping the knife and screaming and... and that memory came much later. I'm sorry. Can I jus... | `19 ([In? p=0.52], [remembered? p=0.55])` | NEEDS_AUDIO_REVIEW [Sliced:1] | 4.80 | 3.04 | 2.97 | 32.3 | 0.90 / 1.69 | 0 / 1 / 1 | Deceptive (โกหก) (`trial_lie_030.mp4`) |
| `anon_ext_10` | `train_ext_18` | We had some drinks at the bar, maybe one... Two? Did you dance? Um, I got onto the dance floor myself, as I explained, um, I've... been a tr... | `19 ([We? p=0.46], [one...? p=0.47 / alt: won])` | CONSISTENT | 4.28 | 4.26 | 1.91 | N/A | 0.91 / 2.58 | 0 / 0 / 0 | Deceptive (โกหก) (`trial_lie_007.mp4`) |
| `anon_ext_11` | `train_ext_18` | My attempt to talk to them was to present a better image of our relationship and downplay the negative aspects. It's not really a big deal, ... | `13 ([bad,? p=0.50], [yeah,? p=0.67])` | CONSISTENT | 4.91 | 2.76 | 2.43 | N/A | 0.68 / 2.62 | 0 / 0 / 0 | Deceptive (โกหก) (`trial_lie_018.mp4`) |
| `anon_ext_12` | `train_ext_18` | Again, it took, it took a long time, um... It took a long time for me to get to this point. I never wanted to. admit to this, and... I had w... | `18 ([um...? p=0.55], [admit? p=0.34])` | CONSISTENT | 4.85 | 1.74 | 2.45 | N/A | 1.03 / 3.45 | 0 / 0 / 0 | Deceptive (โกหก) (`trial_lie_017.mp4`) |
| `anon_ext_13` | `train_ext_18` | I started the investigation simulating how I start all my, um, investigations. I, I have a, uh, what's called a primary briefing, where all ... | `17 ([I? p=0.38], [simulating? p=0.65])` | CONSISTENT | 5.13 | 1.75 | N/A | 58.0 | 4.89 / 5.33 | 1 / 0 / 1 | Truthful (พูดจริง) (`trial_truth_030.mp4`) |
| `anon_ext_14` | `train_ext_18` | Well... what I do recall is when I filled the gas can. rather than... I have just a loose gas hose somewhere. I didn't have anywhere to put ... | `19 ([Well...? p=0.43], [what? p=0.20])` | CONSISTENT | 4.58 | 1.90 | 2.21 | 32.5 | 0.60 / 1.02 | 0 / 0 / 0 | Deceptive (โกหก) (`trial_lie_016.mp4`) |
| `anon_ext_15` | `train_ext_18` | You know, when Rusty backed up out of the garage. So Rusty had a communication with him, but now I know why, but Hemi never responded, becau... | `7 ([You? p=0.12], [Rusty? p=0.46])` | CONSISTENT | 4.49 | 4.90 | 1.91 | N/A | 1.29 / 3.04 | 0 / 0 / 0 | Truthful (พูดจริง) (`trial_truth_005.mp4`) |
| `anon_ext_16` | `train_ext_18` | Oh, I didn't understand my rights. Now, I'm not a coward. You get what I'm saying? All I know is this. My son is eight years old right now. ... | `13 ([Oh,? p=0.04], [Now,? p=0.46])` | CONSISTENT | 5.70 | 3.23 | N/A | N/A | 2.43 / 3.43 | 0 / 0 / 0 | Truthful (พูดจริง) (`trial_truth_026.mp4`) |
| `anon_ext_17` | `train_ext_18` | I do. So what happened? What do you remember that morning? I was, uh, in office and I got a call. uh, maybe... close to 9 o'clock, I don't r... | `33 ([I? p=0.55], [So? p=0.57])` | CONSISTENT | 4.13 | 4.72 | 1.97 | N/A | 0.96 / 3.56 | 1 / 0 / 1 | Truthful (พูดจริง) (`trial_truth_004.mp4`) |
| `anon_ext_18` | `train_ext_18` | So, by who? I have no idea. A police officer, I presume. You'd have to ask my mother or my brother. Did they tell you where Rusty was? Nope.... | `30 ([So,? p=0.64], [who?? p=0.46])` | CONSISTENT | 4.40 | 3.44 | 2.36 | N/A | 1.12 / 3.38 | 0 / 0 / 0 | Truthful (พูดจริง) (`trial_truth_007.mp4`) |

---

## 5. วิเคราะห์ความผิดพลาด (Failure Case Analysis)

1. **False Positives ของ Mode A / Mode C (พยานพูดจริง แต่ระบบเตือนว่าผิดปกติ — 3 คลิป):**
   - **`anon_clip_12` (`trial_truth_059.mp4`, Truthful):** พยานเล่าเรื่องชีวิตคู่และการหย่าร้างด้วยอารมณ์ตึงเครียด ทำให้อัตราการกะพริบตาพุ่งสูงถึง **`72.2 ครั้ง/นาที`** (เทียบกับค่ากลางประชากร 23.3 ± 9.0) ดันค่า `max d_sq/df` ขึ้นไปถึง **`16.609`** นี่คือข้อพิสูจน์ชัดเจนว่า **ความเครียดทางอารมณ์ (Emotional Distress) ของผู้พูดความจริง สร้างสัญญาณ Mahalanobis shift ที่รุนแรงได้ไม่ต่างจากคนโกหก**
   - **`anon_ext_13` (`trial_truth_030.mp4`, Truthful):** มี `blink_rpm = 58.0` และ `art_rate = 5.13 syl/s` ทำให้ `max d_sq/df = 5.329` (และระบบตัด `jitter` เป็น `None` เพราะคุณภาพเสียงไม่ผ่านเกณฑ์)
   - **`anon_ext_17` (`trial_truth_004.mp4`, Truthful):** คลิปยาว 16 เทิร์น (`80.6 วินาที`) มี `f0_sd = 4.72 st` ทำให้มี 1 เทิร์นที่ `d_sq/df = 3.561` เกินเกณฑ์ 3.5 เล็กน้อย
2. **False Negatives ของ Mode B / Mode C (คนโกหก แต่ระบบจับไม่ได้ — 8 คลิปใน Mode C, 11 คลิปใน Mode B):**
   - **`anon_clip_03` (`trial_lie_058.mp4`), `anon_clip_08` (`trial_lie_059.mp4`), `anon_clip_09` (`trial_lie_060.mp4`):** ในคลิปสั้น 20–40 วินาที ผู้พูดตอบคำถามอย่างมั่นใจ (`"One hundred percent"`, `"Never have I been alone with Candace Contee"`) โดยไม่มีข้อความขัดแย้งกับตัวเองภายในช่วงเวลา 30 วินาทีนั้น และค่า `max d_sq/df` อยู่ที่ `1.16–2.15` เท่านั้น
   - **ข้อจำกัดด้านการตรวจจับใบหน้า (`blink_rpm = N/A` ใน 13/30 คลิป):** มุมกล้องศาลที่หันข้างหรือภาพความละเอียดต่ำทำให้ MediaPipe head-pose gate ปฏิเสธการวัด `blink_rpm` เพื่อกันค่าเพี้ยน ส่งผลให้ Mode A เหลือเฉพาะฟีเจอร์เสียงในการคำนวณ `d_sq/df`

---

## 6. ประสิทธิภาพฮาร์ดแวร์ (Latency & Resource Profile)

ทดสอบบน Windows 11 (CPU-only สำหรับ Whisper/MediaPipe/Praat, Process RSS วัดด้วย `psutil 7.2.2`):

| รายการวัดผล | ค่าที่วัดได้จริง |
|---|---|
| **จำนวนคลิปทั้งหมด** | 30 คลิป (ความยาววิดีโอรวม **1,071.59 วินาที** / 17.86 นาที) |
| **เวลาประมวลผลรวม (Wall-Clock)** | **513.56 วินาที** (8.56 นาที) |
| **Real-Time Factor (RTF)** | **0.479x** (~2.09 เท่าของความเร็ว Real-time) |
| **ความเร็วเฉลี่ยต่อวิดีโอ 1 นาที** | **28.75 วินาที / นาทีวิดีโอ** |
| **Peak RAM (Process RSS)** | **1,415.8 MB** (~1.38 GB) |
| **CPU Utilization (Last Sample)** | **350.7%** (multi-thread บน 10 cores / 16 threads) |

### สัดส่วนเวลาแยกตาม Component (Cold vs Warm Cache)
| Component | Cold (`anon_clip_01.mp4`, 34.0s, 450 frames) | Warm (`anon_clip_02.mp4`, 29.0s, 435 frames) | สัดส่วนเวลาหลัก (Warm) |
|---|---:|---:|---:|
| **PyAV Decode (Audio 16k + 15 FPS RGB)** | 0.381s | 0.317s | 3.4% |
| **MediaPipe `FaceLandmarker` (15 FPS)** | 3.709s | 3.077s | 33.0% |
| **Praat + eGeMAPS (5s window)** | 0.125s | 0.047s | 0.5% |
| **Whisper `small.en` (CPU)** | 6.077s | 5.887s | **63.1%** |

---

## 7. ผลอัปเกรด v2: DECODE 97%, แก้ Regression วิดีโอศาล & ทดสอบ Werewolf เต็มเกม (6 เกม)

### 7.1 DECODE v2 (`dev.jsonl` 100 คู่สมดุล: 50 Pos / 50 Neg)
อัปเกรด Stage 1 (`threshold = 0.72`, Clause/Sentence Splitting, Turn-0 Context, `same_speaker_only=True`):

| Pipeline Stage | TP | FP | TN | FN | Precision | Recall | F1-Score | Accuracy | LLM Calls ที่ประหยัดได้ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Stage 1: DeBERTa-v3-xsmall (`>= 0.72`)** | 48 | 35 | 15 | 2 | 0.5783 | **0.9600** | 0.7218 | 0.6300 | 1,787 / 3,135 (57.0%) |
| **Full 2-Stage (DeBERTa + Blind Judge)** | **47** | **0** | **50** | **3** | **1.0000** | **0.9400** | **0.9691** | **0.9700** | **1,787 / 3,135 (57.0%)** |

### 7.2 ผลหลังแก้ Audit 7 จุด + Recalibrate 54 คลิป + 5-Fold CV (`thr=1.8`) ในวิดีโอผสม 24 คลิป
1. คืน `acoustic_location_shrinkage_k = 3.0`, ขยาย `max_frames = 9000` (10 นาที), ใช้ causal rolling 10s `blink_rate_per_min` (`pop_sigma` clamped `[5.0, 12.0]`), และคืน block-diagonal voice correlations
2. Gate ฟีเจอร์ `AU*` และ `eye_gaze_*` (`|yaw| <= 20°`), ตัด `loudness_db` / `au01_inner_brow_raise` / `au25_lips_part` ออกจาก Mahalanobis `df`, และจูนเกณฑ์ `mean(d_sq/df) >= 1.8` ด้วย 5-fold CV บน `Train/` 109 คลิป (`best_sustained_threshold = 1.8`, CV Acc `47.7%` บน `Train/`) โดยไม่แตะ `Test/`

| Dataset Split | Mode | TP | FP | TN | FN | Precision | Recall | F1-Score | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Overall (`n=24`)** | Mode A (Physio) | 1 | 1 | 11 | 11 | 0.5000 | 0.0833 | 0.1429 | 0.5000 |
| **Overall (`n=24`)** | **Mode B (Semantic)** | **3** | **1** | **11** | **9** | **0.7500** | **0.2500** | **0.3750** | **0.5833** |
| **Overall (`n=24`)** | Mode C (Fusion) | 3 | 2 | 10 | 9 | 0.6000 | 0.2500 | 0.3529 | 0.5417 |
| **`RLDD Test/` (`n=12`)** | Mode A (Physio) | 1 | 1 | 5 | 5 | 0.5000 | 0.1667 | 0.2500 | 0.5000 |
| **`RLDD Test/` (`n=12`)** | **Mode B (Semantic)** | **2** | **0** | **6** | **4** | **1.0000** | **0.3333** | **0.5000** | **0.6667** |
| **`RLDD Test/` (`n=12`)** | Mode C (Fusion) | 2 | 1 | 5 | 4 | 0.6667 | 0.3333 | 0.4444 | 0.5833 |
| **`Werewolf` 20s (`n=12`)** | Mode A (Physio) | 0 | 0 | 6 | 6 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| **`Werewolf` 20s (`n=12`)** | Mode B & Mode C | 1 | 1 | 5 | 5 | 0.5000 | 0.1667 | 0.2500 | 0.5000 |

### 7.3 ประเมิน `Werewolf-Among-Us` แบบเต็มเกมต่อเนื่อง (`6 เกม, 26 ผู้เล่น`)
รันต่อเนื่องตั้งแต่ Turn 1 จนจบเกมแบบ **Blind 100%** (`benchmarks/run_werewolf_full_games.py`) โดยสะสม `BaselineStore` รายบุคคล (`n_eff >= 3`) + วิเคราะห์คำอ้าง Role และ Self-Contradiction ข้ามเทิร์นทั้งเกม:

| Mode | TP | FP | TN | FN | Precision | Recall | F1-Score | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Mode A (In-Game Per-Player Baseline)** | 4 | 9 | 9 | 4 | 0.3077 | 0.5000 | 0.3810 | 0.5000 |
| **Mode B (Blind Stage-2 Full-Game Reasoning)** | 4 | 5 | 13 | 4 | 0.4444 | 0.5000 | 0.4706 | **0.6538** |
| **Mode C (Neuro-Symbolic Fusion)** | **6** | 7 | 11 | **2** | **0.4615** | **0.7500** | **0.5714** | **0.6538** |

**ข้อค้นพบสำคัญจากเกมเต็ม:** ในเกม *One Night Ultimate Werewolf* บทบาทฝั่งดีอย่าง `Tanner` มักจงใจโกหกเพื่อล่อให้คนอื่นโหวตตัวเองออก ทำให้เกิด False Positive ใน Mode B (5 คน) แต่เมื่อรวม Mode C เข้ากับ Baseline ต่อเนื่องในเกม กลับช่วยดึง Recall ของฝ่าย Werewolf/Minion จาก **50.0% -> 75.0%** (`TP=6/8`, `F1=0.5714`)
