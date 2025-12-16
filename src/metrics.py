import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from sympy import Eq, simplify, count_ops, Symbol

class DeepMathMetrics:
    def __init__(self):
        # Cấu hình parse mạnh mẽ (hiểu 2x là 2*x)
        self.transformations = (standard_transformations + (implicit_multiplication_application,))

    def compute_all_metrics(self, generated_path_text, ground_truth_value_str):
        """
        Hàm chính để tính toán cả 3 chỉ số cho 1 lời giải (path).
        
        Input:
            generated_path_text: Toàn bộ chuỗi suy luận của model.
            ground_truth_value_str: Đáp án đúng (VD: "5", "x=5").
            
        Output: dict {EE, ASS, TSA}
        """
        # 1. Tiền xử lý: Tách các bước và trích xuất biểu thức toán học
        steps = self._extract_steps_with_math(generated_path_text)
        if not steps:
            return {"EE": 0.0, "ASS": 0.0, "TSA": 0.0}
        
        # Parse Ground Truth thành giá trị số/biểu thức (VD: x=5 -> {x: 5})
        gt_solution = self._parse_ground_truth(ground_truth_value_str)

        # --- Metric 1: Expression Equivalence (EE) ---
        # Kiểm tra tính liên kết logic giữa các bước liền kề
        valid_transitions = 0
        total_transitions = 0
        for i in range(len(steps) - 1):
            expr_curr = steps[i]['expr']
            expr_next = steps[i+1]['expr']

            if expr_curr is None or expr_next is None:
                continue

            total_transitions += 1
            # Kiểm tra: Liệu expr_curr có tương đương expr_next?
            # Lưu ý: Model thường biến đổi phương trình. A=B => C=D.
            # Ta check simplify(curr - next) == 0 (nếu là biểu thức) 
            # hoặc check tập nghiệm (nếu là phương trình)
            if self._check_equivalence(expr_curr, expr_next):
                valid_transitions += 1
        ee_score = (valid_transitions / total_transitions) if total_transitions > 0 else 0.0

        # --- Metric 2: Transformation Step Accuracy (TSA) ---
        # Kiểm tra từng bước có đúng với đáp án thực tế không
        correct_steps = 0
        total_math_steps = 0

        for step in steps:
            expr = step['expr']
            if expr is None:
                continue
            total_math_steps += 1
            if self._check_consistency_with_ground_truth(expr, gt_solution):
                correct_steps += 1

        tsa_score = (correct_steps / total_math_steps) if total_math_steps > 0 else 0.0

        # --- Metric 3: Algebraic Simplification Score (ASS) ---
        # Chỉ tính trên bước cuối cùng (Final Answer)
        last_expr = steps[-1]['expr']
        ass_score = 0.0
        if last_expr is not None:
            ass_score = self._calculate_ass(last_expr)
        
        return {
            "EE": ee_score,
            "ASS": ass_score,
            "TSA": tsa_score
        }
    
    # ================= HELPERS =================#
    def _extract_steps_with_math(self, text):
        """Tách dòng và parse SymPy cho từng dòng"""
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        parsed_steps = []

        for line in lines:
            # Bỏ qua dòng text không có số/toán
            if not any(c.isdigit() for c in line): continue

            # cố gắng parse
            try:
                # Lấy phần toán trong dòng (cơ bản)
                math_part = line.split(":")[-1].strip() # Fix cho dạng "Step 1: x+1=2"
                expr = self._safe_parse(math_part)
                parsed_steps.append({'text': line, 'expr': expr})
            except:
                continue
        return parsed_steps
    
    def _safe_parse(self, text):
        try:
            # Xử lý dấu bằng: "x + 1 = 2" -> Eq(x+1, 2)
            if "=" in text:
                lhs, rhs = text.split("=", 1)
                return Eq(parse_expr(lhs, transformations=self.transformations), 
                          parse_expr(rhs, transformations=self.transformations))
            return parse_expr(text, transformations=self.transformations)
        except:
            return None
        
    def _check_equivalence(self, expr1, expr2):
        """Kiểm tra 2 biểu thức/phương trình có tương đương ngữ nghĩa không"""
        try:
            # Case 1: Cả 2 là phương trình (Eq)
            if isinstance(expr1, Eq) and isinstance(expr2, Eq):
                # Cách đơn giản: Chuyển về dạng A - B = 0 và so sánh
                return simplify((expr1.lhs - expr1.rhs) - (expr2.lhs - expr2.rhs)) == 0
            
            # Case 2: Cả 2 là biểu thức
            return simplify(expr1 - expr2) == 0
        except:
            return False
    
    def _parse_ground_truth(self, gt_str):
        """
        Output: Dictionary nghiệm. VD: "5" -> {x: 5} (nếu biết biến) hoặc giá trị raw.
        Ở đây ta giả định GT là giá trị số hoặc biểu thức đích.
        """
        try:
            return parse_expr(gt_str, transformations=self.transformations)
        except:
            return None
    
    def _check_consistency_with_ground_truth(self, step_expr, gt_value):
        """
        Thay Ground Truth vào phương trình bước hiện tại xem có đúng không.
        VD: Step: '2x = 10', GT: 5. Thay x=5 -> 10=10 (True).
        """
        try:
            if gt_value is None: return False

            # Nếu step là phương trình Eq(lhs, rhs)
            if isinstance(step_expr, Eq):
                # Tìm biến trong phương trình
                symbols = step_expr.free_symbols
                if not symbols: return False # Phương trình hằng số 1=2
                # Thay thế tất cả biến bằng gt_value (Giả sử bài toán 1 biến)
                # Lưu ý: Logic này đúng cho bài toán tìm x.
                # Với bài toán rút gọn, gt_value là kết quả cuối.

                # Thử check: lhs - rhs = 0?
                # Cần subs biến. VD: step: 2*x - 10 = 0. GT: x=5.
                check = step_expr.subs({list(symbols)[0]: gt_value})
                return simplify(check.lhs - check.rhs) == 0
            
            # Nếu step là biểu thức (VD đang rút gọn): 'x + x' -> '2x'
            # Check xem biểu thức này có bằng GT không? (Không khả thi cho bài rút gọn từng bước)
            # Với bài rút gọn: GT là kết quả cuối.
            # Nếu bước hiện tại biến đổi đúng, giá trị của nó với x bất kỳ phải bằng GT? 
            # Không, bài rút gọn thì biểu thức thay đổi hình dạng nhưng giá trị giữ nguyên.
            # -> Check: simplify(step_expr - gt_value) == 0 ?? 
            # (Chỉ đúng nếu GT là biểu thức gốc chưa rút gọn hoặc đã rút gọn).
            
            # Tạm thời implement cho bài toán tìm nghiệm (Solving):
            symbols = step_expr.free_symbols
            if symbols:
                val = step_expr.subs({list(symbols)[0]: gt_value})
                if isinstance(val, Eq): return simplify(val.lhs - val.rhs) == 0

            return False
        except:
            return False
    
    def _calculate_ass(self, expr):
        """
        ASS = 1 - (khoảng cách đến dạng canonical)
        Dùng count_ops để đo độ phức tạp.
        """
        try:
            # Dạng tối giản lý tưởng
            canonical = simplify(expr)
            ops_generated = count_ops(expr)
            ops_canonical = count_ops(canonical)

            if ops_generated <= ops_canonical:
                return 1.0 # Đã tối giản tốt
            
            # Phạt dựa trên độ phức tạp thừa
            # VD: Gen: 2+2 (ops=1). Canon: 4 (ops=0). Diff=1.
            # Score = 1 / (1 + diff)
            return 1.0 / (1.0 + (ops_generated - ops_canonical))
        
        except:
            return 0.0
        