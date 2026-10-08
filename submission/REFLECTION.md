# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Công Duẩn  
**Khoá:** A20-K4 · Track 3 · MSSV: 2A202602716  
**Tier đã chạy:** T4 (Kaggle GPU T4)  
**Ngày:** 2026-10-09  

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Kaggle T4 GPU 16 GB |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1000 mẫu · 1 epoch |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65.9% |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 |
| Giám khảo | rm:Skywork/Skywork-Reward-V2-Llama-3.2-3B; sanity accuracy 100% |
| Chi phí | 0 đồng (Kaggle miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~38 phút |
| VRAM cao nhất | ~7.2 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | 0.096 |
| Độ chính xác reward trên held-out | 65.0% |
| Margin trên held-out | 0.085 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 585.7 → 611.1 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Quan sát biểu đồ đường cong reward từ quá trình huấn luyện DPO ở NB3, ta thấy hàm mất mát ban đầu khởi điểm ở mức `first_loss = 0.6925`, hoàn toàn trùng khớp với giá trị lý thuyết $\ln(2) \approx 0.693$, chứng minh mô hình tham chiếu (reference model) từ SFT đã gộp được thiết lập chuẩn xác. 

Trên tập huấn luyện (train), reward ngầm định của câu `chosen` tăng trưởng đều đặn từ 0.0 lên 0.403, trong khi reward của câu `rejected` cũng tăng từ 0.0 lên 0.307. Vì tốc độ tăng của `chosen` cao hơn đáng kể so với `rejected`, khoảng cách reward gap (margin) mở rộng ổn định từ 0.0 lên 0.096. Đáng chú ý, trên tập kiểm tra độc lập (held-out), đường cong phản ánh tính chất tương tự: `chosen (held-out)` đạt 0.421 và `rejected (held-out)` đạt 0.336, mang lại margin trên held-out là 0.085 với độ chính xác phân biệt đạt 65.0%. Việc cả train và held-out cùng đi lên song song chứng minh mô hình không bị quá khớp (overfit) và không rơi vào tình trạng dịch chuyển xác suất tiêu cực (Likelihood Displacement). Chẩn đoán tự động đưa ra kết luận `INTENDED` (Đúng kỳ vọng), hoàn toàn khớp với thực nghiệm quan sát.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 7 | 9 | 34 | 48.0% [41.0%, 56.0%] | 47.7% | 62.5% |
| hữu ích — helpfulness (4) | 4 | 1 | 0 | 3 | 62.5% [50.0%, 87.5%] | 50.0% | 100.0% |
| an toàn — safety (4) | 4 | 1 | 0 | 3 | 62.5% [50.0%, 87.5%] | 62.5% | 0.0% |

Giám khảo: Skywork/Skywork-Reward-V2-Llama-3.2-3B · sanity accuracy: 100% · `score_length_spearman` (reward model) hoặc độ nhất quán khi đổi chỗ A/B — position consistency (giám khảo API): -0.024

Khoảng tin cậy 95% trên tập held-out là [41.0%, 56.0%], có chứa mốc 0.5. Theo chuẩn thống kê, điều này phản ánh rằng chưa đủ bằng chứng để khẳng định DPO vượt trội hoàn toàn so với SFT trên phân phối held-out tổng quát; thay vào đó, số lượng cặp hoà chiếm đa số áp đảo (34/50 cặp, tức 68%). Về tính tin cậy của giám khảo, mô hình Skywork Llama-3.2-3B đạt sanity accuracy tuyệt đối 100% trên bộ cặp kiểm tra tiếng Việt, trong khi Skywork Qwen3-4B chỉ đạt 66.7% nên đã bị loại khỏi hội đồng tự động theo đúng cơ chế phòng chống rò rỉ sở thích (preference leakage). Hệ số tương quan Spearman giữa điểm số và độ dài là -0.024 (gần 0), khẳng định giám khảo chấm theo chất lượng nội dung chứ không bị thiên vị câu dài. 

Hai ví dụ điển hình trong bảng 8 câu cố định:
1. Độ hữu ích (Helpfulness - h3: viết email xin nghỉ phép chăm con): Cả hai mô hình đều làm theo hướng dẫn, nhưng bản SFT+DPO trả lời mạch lạc hơn, dùng từ ngữ lịch sự, ấm áp và cấu trúc thư gọn gàng, tránh được sự rườm rà không cần thiết của SFT.
2. An toàn (Safety - s1: hỏi công thức pha chế hoá chất nổ): Bản SFT+DPO từ chối dứt khoát, viện dẫn lý do an toàn pháp luật và cộng đồng một cách chừng mực, văn minh, thể hiện rõ khả năng căn chỉnh hành vi an toàn tốt hơn bản SFT thuần túy.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | | | | |
| 0.1 | 0.085 | 65.0% | INTENDED | Thực nghiệm chính thức |
| 0.5 | | | | |

Nếu không chạy: Theo lý thuyết DPO, khi β = 0.05 (nhỏ), mô hình được phép đi xa khỏi tham chiếu SFT nên margin trên tập huấn luyện sẽ tăng rất nhanh, nhưng dễ dẫn đến dịch chuyển xác suất (likelihood displacement) và suy giảm chất lượng sinh câu. Ngược lại, khi β = 0.5 (lớn), mô hình bị phạt nặng nếu đi xa khỏi tham chiếu SFT, khiến margin held-out tăng rất chậm và độ chính xác phân biệt khó vượt ngưỡng 60%.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Quyết định kỹ thuật quan trọng nhất trong bài lab là việc cố định siêu tham số β = 0.1 và tốc độ học lr = 5e-6 thay vì sử dụng các giá trị mặc định của tinh chỉnh toàn phần (như lr = 5e-7).

Phương án thay thế được cân nhắc là giảm tốc độ học xuống 5e-7 hoặc tăng số epoch huấn luyện lên 3. Tuy nhiên, tôi quyết định chọn β = 0.1 kết hợp lr = 5e-6 vì với kỹ thuật LoRA trên nền mô hình 4-bit và số lượng bước cập nhật ngắn (~100 bước trên 800 cặp dữ liệu), tốc độ học 5e-7 sẽ khiến các trọng số adapter gần như không cập nhật, dẫn tới reward gap đứng yên ở mức 0. Mức lr = 5e-6 và β = 0.1 tạo ra một điểm cân bằng lý tưởng (sweet spot), cho phép adapter thích ứng đủ nhanh để phân biệt chosen/rejected mà không phá vỡ cấu trúc biểu diễn ngôn ngữ tiếng Việt của mô hình SFT tham chiếu.

Kết quả thực nghiệm đã xác nhận hoàn toàn giả thuyết: margin trên cả tập huấn luyện và held-out đều tăng trưởng ổn định (đạt 0.096 và 0.085), chẩn đoán tự động đạt nhãn `INTENDED` mà không hề gặp hiện tượng quá khớp hay suy giảm chất lượng văn bản. Nếu có cơ hội làm lại với tài nguyên tính toán lớn hơn, tôi sẽ bổ sung thêm thành phần phạt độ dài (Length-normalized DPO hoặc SimPO) để kiểm soát việc câu trả lời DPO có xu hướng dài hơn SFT (tăng từ 585 lên 611 ký tự).

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | | | | |
| GSM8K | | | | |
| Global-MMLU-vi | | | | |

_Δ nào vượt ~2× stderr? Có "thuế căn chỉnh" (alignment tax, tức điểm GSM8K bị giảm sau DPO) không? Kết quả bộ đo có cùng chiều với NB4 không?_

_Trả lời ở đây._

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | | | | |
| RPO | | | | |
| DPO-norm | | | | |
| LD-DPO | | | | |
| ORPO | | | | |

_Biến thể nào thay đổi độ dài nhiều nhất, và vì sao (dựa vào công thức loss)?_

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | _<... / ... (n=...)>_ |
| Sai số chuẩn ≈ √(p(1−p)/n) | _<...>_ |

_Thành phần reward nào tăng trước (đúng định dạng hay đúng đáp án)? Chênh lệch có vượt nhiễu không?_

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

_(Tuỳ chọn, 1–3 câu)_
