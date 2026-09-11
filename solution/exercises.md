# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi tăng temperature từ 0.0 lên 1.5, độ ngẫu nhiên và tính sáng tạo trong việc lựa chọn token tăng rõ rệt. Ở mức 0.0 và 0.5, mô hình phản hồi mang tính tất định (deterministic), tập trung vào các sự thật phổ biến (như xuất khẩu cà phê Robusta hay chiều dài bờ biển) với cấu trúc câu chuẩn mực, lặp lại ổn định. Ở mức 1.0 và 1.5, câu trả lời phong phú, góc nhìn độc đáo và văn phong bay bổng hơn (hang Sơn Đoòng, văn hóa cà phê trứng), tuy nhiên ở mức 1.5 câu chữ bắt đầu có xu hướng phóng đại và tiềm ẩn nguy cơ xuất hiện ảo giác (hallucination).

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt temperature trong khoảng từ 0.1 đến 0.3 (tối đa không quá 0.5). Lý do là chatbot chăm sóc khách hàng ưu tiên hàng đầu tính chính xác, tính nhất quán và độ tin cậy của thông tin (chính sách đổi trả, giá sản phẩm, quy trình kỹ thuật), tránh tuyệt đối việc mô hình suy diễn sai lệch hoặc bịa đặt dữ liệu (hallucination). Mức nhiệt độ thấp giúp phân phối xác suất tập trung vào các token chuẩn xác nhất nhưng vẫn giữ được độ trôi chảy, tự nhiên trong giao tiếp.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Xét theo bảng giá output, GPT-4o ($0.010/1K token) đắt gấp khoảng 16.7 lần so với GPT-4o-mini ($0.0006/1K token). Với workload 30.000 lượt gọi/ngày tương đương 10,5 triệu output token/ngày, chi phí output của GPT-4o là $105/ngày (~$3.150/tháng), trong khi GPT-4o-mini chỉ tốn $6.30/ngày (~$189/tháng). Trường hợp GPT-4o hoàn toàn xứng đáng: các bài toán đòi hỏi suy luận phức tạp, tư vấn pháp lý, phân tích mã nguồn chuyên sâu hoặc chẩn đoán y tế cần độ chính xác tối đa. Trường hợp nên dùng GPT-4o-mini: chatbot hỏi đáp FAQ thường gặp, phân loại ý định người dùng (intent classification), tóm tắt tin tức ngắn hoặc trích xuất thực thể số lượng lớn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Hai phản hồi có sự phân hóa sâu sắc về phong cách, độ sâu và từ vựng. Với vai giáo viên tiểu học, phản hồi sử dụng câu ngắn, giọng điệu gần gũi, ẩn dụ blockchain như cuốn sổ ghi chép chung của cả lớp mà ai cũng nhìn thấy và không thể tự ý sửa đổi. Ngược lại, vai chuyên gia tài chính sử dụng thuật ngữ chuyên sâu như sổ cái phân tán (DLT), hàm băm mật mã học (cryptographic hash), cơ chế đồng thuận PoW/PoS và tính bất biến (immutability). System prompt hoạt động như cơ chế điều hướng (steering constraint), định hình không gian ngữ nghĩa và xác định phong cách diễn đạt cho mô hình xuyên suốt quá trình sinh nội dung.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Thử nghiệm với đoạn văn tiếng Việt 93 từ: số token thực tế đo bằng tiktoken là 119 token, trong khi công thức ước lượng thô cho ra 124.0 token (chênh lệch khoảng 4.0%). Tiếng Việt tốn nhiều token hơn tiếng Anh cùng độ dài vì các mô hình mã hóa Byte-Pair Encoding (BPE) của OpenAI được huấn luyện chủ yếu trên ngữ liệu tiếng Anh, nơi mỗi từ phổ biến thường là 1 token. Tiếng Việt là ngôn ngữ đa âm tiết, sử dụng bảng chữ cái Latinh kết hợp dấu thanh và ký tự đặc biệt (ă, â, đ, ê, ô, ơ, ư); khi mã hóa UTF-8, các ký tự có dấu chiếm từ 2-3 byte khiến tokenizer phải phân rã một từ tiếng Việt thành nhiều sub-token hoặc từng byte riêng lẻ.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming mang ý nghĩa sống còn trong các ứng dụng đối thoại tương tác trực tiếp với người dùng (như chatbot, trợ lý CLI, coding copilot), nơi chỉ số Time-to-First-Token (TTFT) quyết định cảm giác phản hồi nhanh nhạy; người dùng được đọc nội dung tức thì ngay khi từng chunk được sinh ra thay vì phải chờ đợi toàn bộ câu trả lời hoàn tất. Ngược lại, non-streaming phù hợp hơn cho các tiến trình xử lý ngầm (background batch processing), các tác vụ sinh dữ liệu cấu trúc (JSON schema để gọi API/tool use), hoặc khi hệ thống bắt buộc phải kiểm duyệt nội dung (moderation/safety filter) toàn bộ phản hồi trước khi gửi về client.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giúp giảm theo cấp số nhân tần suất gửi request của client lên máy chủ gặp sự cố (chờ 0.1s, 0.2s, 0.4s...), tạo ra khoảng thời gian nghỉ đủ dài để server giải tỏa hàng đợi và tự phục hồi. Nếu hàng nghìn client cùng retry với một mức delay cố định (ví dụ đúng 1 giây), toàn bộ các client sẽ đồng loạt gửi lại request vào cùng các mốc thời gian tuần hoàn, gây ra hiện tượng bão yêu cầu (Thundering Herd Problem / Retry Storm). Đợt sóng request đồng thời này sẽ tiếp tục đánh sập server đang nghẽn thay vì giúp hệ thống ổn định trở lại.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona được chọn: "Bạn là trợ giảng AI thông minh, nhiệt tình của khóa học lập trình. Hãy giải thích các khái niệm kỹ thuật một cách trực quan, ngắn gọn bằng tiếng Việt, luôn kèm theo một ví dụ minh họa thực tế dễ hiểu." Việc chỉ định 'bằng tiếng Việt' đảm bảo ngôn ngữ phản hồi đồng nhất, tránh việc model tự động chuyển sang tiếng Anh khi gặp các thuật ngữ lập trình. Yêu cầu 'ngắn gọn' và 'kèm ví dụ thực tế' giúp tối ưu trải nghiệm đọc trên giao diện dòng lệnh (CLI), tiết kiệm token/chi phí và giúp người học nắm bắt bản chất khái niệm ngay lập tức.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất của trợ lý hiện tại là cơ chế trượt cửa sổ cố định 3 lượt (tối đa 6 message), khiến mô hình bị mất trí nhớ dài hạn (context loss) và quên đi các thông tin người dùng đã chia sẻ ở các lượt trò chuyện trước đó. Để cải thiện, có thể triển khai kỹ thuật Tóm tắt ngữ cảnh định kỳ (Conversation Summary Memory): khi lịch sử vượt quá 3 lượt, ứng dụng gọi một model nhẹ (như gpt-4o-mini) để tóm lược ngắn gọn các sự kiện quan trọng trong các lượt cũ, sau đó lưu bản tóm tắt này vào đầu prompt (system message) kết hợp cùng 2-3 lượt trò chuyện gần nhất. Điều này giúp trợ lý nhớ xuyên suốt phiên chat mà vẫn kiểm soát tốt lượng token input.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
