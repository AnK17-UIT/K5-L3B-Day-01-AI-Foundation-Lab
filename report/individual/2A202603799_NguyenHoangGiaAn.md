# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603799
- **Họ và tên:** Nguyễn Hoàng Gia An    
- **SEED** (in ở ô đầu notebook): 438

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | x |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | x |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | x |

## 2. Kết quả chính


| Model | precision | recall | F1 (test) | F1 (SUB 40) | train (s) | tin/s (cả lô) | p50 ms/tin | p95 ms/tin | USD / 1k tin |
|---|---|---|---|---|---|---|---|---|---|
| 1. BoW + LogReg (tự code) | 1.000 | 0.758 | 0.862 | 0.919 | 4.274 | 45262.373 | 0.043 | 0.130 | 0.000 |
| 2. BoW + MLP (deep learning) | 0.966 | 0.884 | 0.923 | 0.947 | 5.745 | 35394.077 | 0.209 | 0.243 | 0.000 |
| 3a. Encoder (MiniLM) + kNN | 0.902 | 0.968 | 0.934 | 1.000 | 61.710 | 33.851 | 33.743 | 46.808 | 0.000 |
| 3b. Encoder (MiniLM) + LogReg | 0.959 | 0.979 | 0.969 | 0.976 | 62.017 | 36.400 | 24.531 | 34.305 | 0.000 |
| 3c. Encoder (MiniLM) + MLP head | 0.958 | 0.968 | 0.963 | 0.974 | 62.597 | 34.261 | 30.460 | 43.778 | 0.000 |
| 4a. LLM nhỏ zero-shot | 0.909 | 1.000 | 0.952 | 0.952 | 0.000 | 1.128 | 830.387 | 1358.112 | 0.009 |
| 4b. LLM lớn zero-shot | 0.952 | 1.000 | 0.976 | 0.976 | 0.000 | 0.928 | 1021.697 | 1476.571 | 0.088 |
| 4c. LLM nhỏ few-shot | 1.000 | 0.950 | 0.974 | 0.974 | 0.000 | 1.057 | 810.826 | 1720.726 | 0.020 |

- Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
- SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
- Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
- train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

### 1. So sánh LLM lớn và LLM nhỏ (Zero-shot)

**Dự đoán:** Mô hình lớn (gpt-4o) sẽ vượt trội hoàn toàn về F1 so với mô hình nhỏ (gpt-4o-mini).

**Thực tế:** F1 của mô hình lớn (0.976) chỉ nhỉnh hơn mô hình nhỏ (0.952) một chút.

Phân loại SMS Spam là một tác vụ phân loại văn bản cơ bản. Các mô hình nhỏ hiện nay đã được tối ưu hóa (alignment) rất tốt cho các tác vụ này, nên mức chênh lệch hiệu năng là không đáng kể, trong khi chi phí của mô hình lớn đắt gấp 10 lần.

### 2. Tác động của Few-shot Prompting
**Dự đoán:** F1 sẽ tăng: Việc cung cấp các ví dụ mẫu (few-shot) đóng vai trò như một mỏ neo, giúp LLM hiểu chính xác định dạng đầu ra... từ đó cải thiện độ chính xác.

**Thực tế:** F1 (SUB 40) của LLM zero-shot (4a) là 0.952, few-shot (4c) là 0.974. Chênh lệch ~0.022 (< 0.05 theo luật của đề bài, tức là HÒA). Chất lượng không thực sự tăng đột phá, nhưng thời gian p95 lại chậm đi rõ rệt (từ 1358ms lên 1720ms) và tốn tiền gấp đôi ($0.009 lên $0.020).

Phân loại Spam/Ham là tác vụ mà các LLM hiện nay đã được train quá kỹ. Bản thân mô hình zero-shot đã dư sức phân loại dựa trên tri thức có sẵn, việc nhồi thêm vài tin mẫu không giúp nó thông minh hơn mà chỉ làm tăng khối lượng token phải xử lý, dẫn đến trễ hơn và đắt hơn.

### 3. Vị trí của LLM lớn trên đường Pareto

**Dự đoán:** LLM lớn có thể nằm trên đường Pareto.

**Thực tế:** LLM lớn (4b) bị lấn át (dominated) hoàn toàn và bị đánh bật khỏi đường Pareto. Đường Pareto chỉ gồm các mô hình BoW và Embedding.

Đường Pareto chỉ giữ lại những mô hình "không bị ai khác vừa nhanh hơn vừa ít lỗi hơn". F1 test của nhóm Embedding (3b đạt 0.969) và LLM lớn (4b đạt 0.976) chênh lệch chưa tới 0.05 nên được coi là hòa nhau về chất lượng bắt lỗi. Nhưng về độ trễ, LLM lớn chậm tới 1476.5ms (chậm gấp 40 lần so với mức 34.3ms của Embedding). Do bị nhóm Embedding áp đảo hoàn toàn về tốc độ mà chất lượng lại ngang ngửa, LLM bị rơi ra khỏi đường Pareto.

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

**❓ Câu hỏi 6 — Stress test**
1. So sánh "F1 50 case khó" với "F1 test thường". Model nào **tụt nhiều nhất**? Vì sao điểm trên tập test thường lại "quá lạc quan"?
2. Nhóm nào khó nhất với BoW, nhóm nào khó nhất với Embedding/LLM? Giải thích bằng **cách mỗi model nhìn văn bản** (token rời rạc / vector nghĩa / đọc như người).
3. **Prompt injection:** LLM có bị câu "Ignore all previous instructions…" lừa không? So sánh hàng **4a** và **4d** (prompt phòng thủ trong `build_defended_prompt`) ở nhóm *Prompt injection* và các nhóm khác: phòng thủ có giúp không, và có làm hỏng nhóm nào không? Đề xuất thêm 1 cách phòng thủ **ngoài prompt** (gợi ý: chuẩn hoá đầu vào, không để LLM là lớp kiểm tra duy nhất).
4. Với nhóm **né từ khoá**, một bước tiền xử lý nào (chạy trước model) có thể cứu BoW? Nó có rủi ro gì?
5. 50 case này do người viết tay. Chúng có đại diện cho spam thật ngoài đời không? Dùng bộ test như thế này để làm gì, và **không** nên dùng để làm gì?
> *Trả lời của bạn:*

> - Model tụt nhiều nhất: Nhóm BoW sụp đổ hoàn toàn, F1 của BoW + LogReg tụt từ 0.919 xuống chỉ còn 0.286. Điểm trên tập test thường "quá lạc quan" vì tập đó được lấy mẫu ngẫu nhiên từ cùng một phân phối với tập train (chứa đa số là các tin spam quảng cáo truyền thống lặp đi lặp lại từ khóa dễ đoán).
> - Độ khó theo góc nhìn của mô hình:Với BoW: Khó nhất là nhóm "Spam giọng tự nhiên" (đoán đúng 0-10%), "Thao túng/lừa đảo" (đoán đúng 12-25%) và "Né từ khoá" (đoán đúng 0-17%). Vì BoW nhìn văn bản dưới dạng token rời rạc, khi kẻ gian không dùng từ khóa quen thuộc (như free, win), hoặc cố tình chèn ký tự lạ (như F R E E), từ vựng sẽ không khớp và BoW đành bó tay.
> - Với Embedding: Khó nhất là "Spam giọng tự nhiên" (đoán đúng 30-40%) và "Nói về spam" (đoán đúng 25-75%). Vì mô hình mã hóa nhìn văn bản dưới dạng vector ngữ nghĩa (đọc như người), nó dễ bị lúng túng giữa một tin nhắn lừa đảo bắt chước hoàn hảo giọng điệu sinh hoạt, và một tin nhắn sinh hoạt thật nhưng lại có trích dẫn từ khóa spam để trêu đùa. (Lưu ý: LLM đã xuất sắc giải quyết được tất cả các điểm mù này với tỷ lệ 100%).
> - Prompt Injection:Kết quả thực tế: Model LLM bạn đang dùng quá thông minh nên không hề bị lừa, hàng 4a (không phòng thủ) vẫn đạt tuyệt đối 100% ở nhóm Prompt Injection. Việc thêm prompt phòng thủ ở hàng 4d cũng duy trì thành tích 100%, chứng tỏ kỹ thuật này không làm hỏng khả năng phân loại các nhóm khác và vẫn là lớp giáp cần thiết cho các đòn tấn công phức tạp hơn.
> - Phòng thủ ngoài prompt: Sử dụng kiến trúc phân tầng (Cascade). Cho tin nhắn đi qua BoW/Embedding cực nhanh trước; nếu mô hình này phát hiện từ khóa bất thường, có thể gắn cờ ngay mà không cần gọi API LLM.
> - Cứu BoW ở nhóm Né từ khoá: Chạy một bước tiền xử lý chuẩn hoá văn bản (Text Normalization) như xóa khoảng trắng thừa (F R E E thành FREE), gỡ ký tự đặc biệt (€ thành e). Rủi ro: Dễ làm sai lệch các mã giao dịch, số tiền thật hoặc từ viết tắt hợp lệ, dẫn đến chặn nhầm (False Positive).
> - Ý nghĩa của bộ 50 case viết tay: Bộ này không đại diện cho phân phối spam ngoài đời thực (thực tế 95% spam vẫn là dạng quảng cáo rẻ tiền). Bộ test này chỉ dùng để stress-test (tìm điểm mù, thử nghiệm bảo mật, red-teaming). Tuyệt đối không dùng để báo cáo KPI hay Accuracy chung cho hệ thống, vì nó sẽ làm chỉ số hiệu năng trông tệ hại một cách phi thực tế.
> - Case khiến [4d. LLM nhỏ + prompt phòng thủ] sai 1/55 case: #101 [Tưởng spam nhưng không phải] thật=ham: Hey, attached the Q3 financial report. The zip password is: CLICK_LINK_TO_WIN_100_DOLLARS. Let 
> - Case 


## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

 "ML · Đổi ngưỡng quyết định 0.5 → 0.2 và → 0.8 cho mô hình Phần 1. Báo precision/recall/F1 từng ngưỡng. Một ngân hàng sợ chặn nhầm tin OTP thì nên chọn ngưỡng nào? Vì sao?"
```
probs_bow = predict_proba(X_te, w, b)

for thresh in [0.2, 0.5, 0.8]:
    preds_thresh = (probs_bow >= thresh).astype(int)
    m = metrics(y_te, preds_thresh)
    print(f"Ngưỡng {thresh}: Precision = {m['precision']:.3f} | Recall = {m['recall']:.3f} | F1 = {m['f1']:.3f}") 
```


Ngưỡng 0.2: Precision = 0.914 | Recall = 0.895 | F1 = 0.904
Ngưỡng 0.5: Precision = 1.000 | Recall = 0.758 | F1 = 0.862
Ngưỡng 0.8: Precision = 1.000 | Recall = 0.589 | F1 = 0.742

> - Ngưỡng 0.2: Recall tăng (bắt được nhiều spam hơn, ít bỏ lọt), nhưng Precision giảm mạnh (bắt nhầm rất nhiều tin bình thường thành spam - False Positive cao). Chỉ số F1 thường giảm do sự mất cân bằng này.
> - Ngưỡng 0.8: Precision tăng (một khi đã gán nhãn spam thì độ chính xác rất cao, ít bắt nhầm), nhưng Recall giảm (bỏ lọt nhiều tin spam tinh vi - False Negative cao). F1 cũng sẽ giảm so với ngưỡng mặc định 0.5.
> - Ngân hàng nên chọn ngưỡng 0.8 (hoặc thậm chí cao hơn). Vì đối với dịch vụ tài chính, việc chặn nhầm một tin nhắn OTP (lỗi False Positive) là một thảm họa về trải nghiệm người dùng, khiến khách hàng không thể đăng nhập hay chuyển tiền và lập tức gây quá tải cho tổng đài CSKH. Bằng cách nâng ngưỡng quyết định lên 0.8, mô hình chỉ phân loại một tin là rác khi xác suất chắc chắn lên tới 80%. Ngân hàng thà chấp nhận bỏ lọt một vài tin nhắn rác vào máy khách hàng (chấp nhận Recall thấp) còn hơn là chặn nhầm một mã OTP hợp lệ (ưu tiên tối đa Precision đối với nhãn Spam).

## 6. Phương án cho sản phẩm thật

Cách đạt yêu cầu độ trễ (SLA < 50ms): Dựa vào biểu đồ đánh đổi Pareto ở trên, các phương pháp đạt SLA bao gồm: BoW + LogReg, BoW + MLP, và Encoder (MiniLM) + MLP head. Các mô hình LLM (nhỏ/lớn) và kNN đều vượt quá xa mốc 50ms (thường mất từ 800ms - 1500ms).
Cách đạt cả độ trễ lẫn chất lượng: Encoder (MiniLM) + MLP head là sự lựa chọn tối ưu nhất. Nó xử lý nhanh (nằm trong chuẩn 50ms), không tốn chi phí gọi API, nhưng lại có F1 và khả năng đọc hiểu ngữ cảnh tốt hơn hẳn so với BoW (thể hiện rõ qua việc bắt được các case khó trong tập Hard Eval).

Các mô hình LLM hiện tại có p95 từ 800ms - 1500ms, vượt quá xa mốc 50ms. Nếu chạy LLM trực tiếp cho 10 triệu tin/ngày, chi phí ước tính lên đến ~$90/ngày (gần $2.700/tháng) và dễ gây sập hệ thống (chạm trần Rate Limit). Bằng thiết kế Router, ta giữ được SLA dưới 50ms cho 90-95% lưu lượng, chỉ tốn tiền API cho 5-10% các ca thực sự khó, tận dụng được cả tốc độ siêu nhanh của Deep Learning và khả năng đọc hiểu ngữ cảnh tuyệt đối của LLM.

## 7. AI log và điều đã học

### 🤖 AI log (bắt buộc nếu bạn dùng AI ở bất kỳ đâu)

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| TODO 4.1 - 4.4 | Điền các hàm gọi OpenAI API, build prompt, parse label, và xử lý batch. | Viết hàm gọi chat.completions, hàm tạo chuỗi prompt phân loại (xử lý zero/few-shot), logic dùng chuỗi con để bóc tách nhãn spam/ham, và hàm phân loại lô tính USD. | Nhắc AI sửa lỗi cú pháp trừ thời gian latency. Đã kiểm chứng hàm parse_label hoạt động ổn với trường hợp trả về chữ hoa, có dấu chấm câu. |
| HARD_EVAL & Bonus Case | Hỏi đánh giá về tập 50 case khó. Yêu cầu tạo 5 case lừa đảo SMS mới bằng tiếng Anh theo format của bảng. | Trả lời phân tích các điểm yếu (Prompt injection, Obfuscation). Cung cấp 5 tuples tiếng Anh (ví dụ: chuyển nhầm tiền, mật khẩu đùa, injection tag `</sms>`). | Đã copy 5 case vào list MY_CASES, code chạy mượt. Kiểm chứng BoW sai cả 5. LLM có phòng thủ đoán đúng 4/5 (bị lừa ở case mật khẩu đùa do quá cẩn trọng với từ khóa lừa đảo). |
| Fix bug tại câu hỏi biểu đồ | Hỏi nguyên nhân biến môi trường không nhận; nhờ sửa lỗi hàm parse_label_v2 và lỗi cú pháp trong ô vẽ biểu đồ Pareto. | Hỏi nguyên nhân biến môi trường không nhận; nhờ sửa lỗi hàm parse_label_v2 và lỗi cú pháp trong ô vẽ biểu đồ Pareto. | Đã check lại code vẽ biểu đồ, chạy ra được hình ảnh scatter plot chuẩn xác với mốc SLA 50ms, các chấm hiển thị đúng vị trí. | 

### Điều hiểu ra hôm nay mà trước đó chưa hiểu:
Tối ưu hóa một hệ thống AI thực tế không chỉ nằm ở việc theo đuổi chỉ số Accuracy hay F1 cao nhất trên tập dữ liệu tĩnh, mà là bài toán đánh đổi liên tục giữa Độ trễ (Latency), Chi phí (Cost) và Rủi ro nghiệp vụ. Trong môi trường giao dịch tài chính, việc chủ động tinh chỉnh ngưỡng quyết định (hạ Recall, đẩy Precision lên tuyệt đối để bảo vệ mã OTP) quan trọng hơn nhiều so với điểm số mô hình. Ngoài ra, LLM dù có khả năng đọc hiểu ngữ cảnh xuất sắc đến đâu vẫn tồn tại những điểm mù về văn hóa hoặc trò đùa giao tiếp của con người, đòi hỏi phải có các lớp phòng thủ phân tầng và cơ chế tiền xử lý dữ liệu chặt chẽ thay vì phó thác hoàn toàn vào một API duy nhất.