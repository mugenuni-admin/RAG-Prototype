from fpdf import FPDF
import datetime

class PDF(FPDF):
    def __init__(self, watermark_text=""):
        super().__init__()
        self.watermark_text = watermark_text

    def header(self):
        # Add watermark on every page
        self.set_font("helvetica", "B", 30)
        self.set_text_color(220, 220, 220) # Light gray
        
        # Position it somewhat diagonally
        with self.rotation(angle=45, x=105, y=148):
            w = self.get_string_width(self.watermark_text)
            self.text(x=105 - w/2, y=148, txt=self.watermark_text)

def generate_watermarked_pdf(question, answer, user_email):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    watermark_text = f"CONFIDENTIAL - {user_email} - {timestamp}"
    
    pdf = PDF(watermark_text=watermark_text)
    pdf.add_page()
    
    pdf.set_font("helvetica", "B", 16)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "Data Room Query Result", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(10)
    
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "Question:", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("helvetica", "", 12)
    
    # Replace unicode characters that might break fpdf's standard font
    clean_question = str(question).encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, clean_question)
    pdf.ln(5)
    
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "Answer:", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("helvetica", "", 12)
    clean_answer = str(answer).encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, clean_answer)
    
    # Return as bytes for Streamlit download
    return bytes(pdf.output())
