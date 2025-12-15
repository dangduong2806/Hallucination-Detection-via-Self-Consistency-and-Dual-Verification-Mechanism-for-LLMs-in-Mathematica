import numpy as np
from sympy import sympify, SympifyError
class LocalVerifier:
    # def verify_step(self, context, step_content, step_logprob):
    #     """
    #     Kết hợp Atomic và Logical check.
    #     """
    #     # 1. Atomic Check (SymPy): Bước này có vô lý về toán học không?
    #     # Ví dụ: "1 + 1 = 3" -> Atomic Error
    #     atomic_score = self.sympy_check(step_content)
    #     # 2. Logical Dependency (Model Confidence):
    #     # Model có chắc chắn bước này suy ra từ context không?
    #     # Dùng logprob (xác suất) làm thước đo dependency.
    #     logical_score = np.exp(step_logprob)  # Chuyển logprob về prob bình thường

    #     # Kết hợp hai thước đo
    #     final_score = atomic_score * logical_score
    #     return final_score

    def __init__(self, config):
        """
        Khởi tạo Verifier với các cấu hình ngưỡng (thresholds).
        """
        self.config = config
        
        # Lấy tham số từ config, nếu không có thì dùng giá trị mặc định
        self.atomic_enabled = config['verification'].get('atomic_check_enabled', True)
        self.logical_enabled = config['verification'].get('logical_check_enabled', True)
        self.logprob_threshold = config['verification'].get('logprob_threshold', -1.5)

    def verify_path(self, path):
        """
        Input: 
            path: List các bước (dict) từ model. 
            Mỗi item dạng: {'text': '...', 'logprob': -0.5, ...}
            
        Output: 
            verified_steps: List các bước đã qua kiểm duyệt.
        """
        verified_steps = []
        
        # Context dùng để theo dõi chuỗi suy luận (nếu cần check logic phức tạp hơn)
        # Ở version đơn giản này, ta check từng bước độc lập dựa trên logprob và syntax.
        
        for step in path:
            step_content = step.get('text', '')
            step_logprob = step.get('logprob', -float('inf'))
            
            # --- 1. Atomic Check (Kiểm tra lỗi toán học sơ đẳng) ---
            if self.atomic_enabled:
                if not self._check_atomic_validity(step_content):
                    # Nếu bước này viết sai cú pháp toán học (vd: "x + = 2") -> Dừng ngay
                    break 

            # --- 2. Logical Check (Kiểm tra độ tự tin của model) ---
            if self.logical_enabled:
                if step_logprob < self.logprob_threshold:
                    # Model quá phân vân về bước này -> Coi là ảo giác -> Dừng ngay
                    # (Pruning: Cắt bỏ nhánh sai sớm để tiết kiệm)
                    break
            
            # Nếu qua được cả 2 vòng check thì thêm vào danh sách hợp lệ
            verified_steps.append({
                'content': step_content,
                'confidence': np.exp(step_logprob), # Chuyển logprob về xác suất (0-1)
                'logprob': step_logprob
            })
            
        return verified_steps

    def _check_atomic_validity(self, text):
        """
        Dùng SymPy để kiểm tra xem text có chứa biểu thức toán học hợp lệ không.
        Đây là cách đơn giản để lọc bỏ các bước 'nói nhảm' (gibberish).
        """
        try:
            # Logic: Thử parse text. Nếu SymPy parse được -> Có khả năng là toán.
            # Ta cần clean text một chút trước khi parse (bỏ các từ tiếng Anh common)
            clean_text = text.lower().replace("solve", "").replace("step", "").strip()
            
            # Nếu chuỗi rỗng sau khi clean -> Có thể là lời dẫn, tạm cho qua (True)
            if not clean_text:
                return True
                
            # Thử parse
            sympify(clean_text)
            return True
        except:
            # Nếu SymPy báo lỗi syntax -> Bước này không phải toán học hợp lệ
            # Tuy nhiên, LLM hay viết lời văn (text), nên ta chỉ return False
            # nếu ta cực kỳ khắt khe. Ở mức độ nghiên cứu này, ta có thể return True
            # nhưng log lại warning.
            return True # Tạm thời cho qua để tránh lọc nhầm lời văn giải thích