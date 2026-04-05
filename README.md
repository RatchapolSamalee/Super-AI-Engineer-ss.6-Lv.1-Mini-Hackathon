# SuperAI Engineer Season 6 : Mini Hackathon Portfolio

**ผู้แข่งขัน** : นายรัชพล สม่าหลี (รหัส 600659)

สรุปผลงานจากการแข่งขัน Mini Hackathonภายใต้โครงการ SuperAI Engineer Season 6 ทั้ง 4 Hackathon รวม 5 โจทย์ 

---

## รายการ Hackathon ที่ได้ลงแข่ง

- [ข้อ 1 : Mini Hackathon 1 - Data (วิเคราะห์ข้อมูลการใช้ไฟฟ้า)](#ข้อ-1--mini-hackathon-1---data-วิเคราะห์ข้อมูลการใช้ไฟฟ้า)
- [ข้อ 2 : Mini Hackathon 2 - OCR (สกัดผลเลือกตั้งจากภาพเอกสาร)](#ข้อ-2--mini-hackathon-2---ocr-สกัดผลเลือกตั้งจากภาพเอกสาร)
- [ข้อ 3 : Mini Hackathon 3 - FahMai (ระบบ RAG ถาม-ตอบสินค้า)](#ข้อ-3--mini-hackathon-3---fahmai-ระบบ-rag-ถาม-ตอบสินค้า)
- [ข้อ 4 : Mini Hackathon 4 - Sleep Stage Classification](#ข้อ-4--mini-hackathon-4---sleep-stage-classification)
- [ข้อ 5 : Mini Hackathon 4 - Word Segmentation](#ข้อ-5--mini-hackathon-4---word-segmentation)

---

## ข้อ 1 : Mini Hackathon 1 - Data (วิเคราะห์ข้อมูลการใช้ไฟฟ้า)

### โจทย์

วิเคราะห์ชุดข้อมูลการใช้ไฟฟ้ารายเดือนของ 18 เขตให้บริการของการไฟฟ้านครหลวง (MEA) ตั้งแต่ปี พ.ศ. 2562-2568 (83 แถว) โดยต้องทำความสะอาดข้อมูล (Data Cleaning), สำรวจข้อมูลเชิงลึก (EDA) และตอบคำถามเชิงวิเคราะห์ 3 ข้อ พร้อมให้ข้อเสนอแนะเชิงนโยบาย

โจทย์นี้เน้น Data Analytics / EDA เป็นหลัก ไม่ได้มีการใช้โมเดล AI โดยตรง แต่ใช้เครื่องมือทางสถิติ และ Visualization ในการหาคำตอบ

### Preprocessing (Data Cleaning)

ข้อมูลที่ดึงมาจาก Thackle มีหลายจุดให้ทำความสะอาด ดังนี้ :

- ตรวจสอบคอลัมน์ `(หักไฟทำการ)` พบว่าเป็น dtype object ไม่ใช่ตัวเลข จึงแปลงเป็น float แล้วเทียบกับ sum ของทุกเขตสรุปว่าเป็นยอดรวม จึงเปลี่ยนชื่อเป็น `รวม` และลบคอลัมน์เก่าทิ้ง
- เติมค่าว่างในคอลัมน์ `ปี` จำนวน 76 แถว ด้วย Forward Fill เนื่องจากข้อมูลต้นฉบับระบุปีแค่แถวแรกของแต่ละปี ทำให้แถวที่เหลือเป็นค่าว่าง
- ลบอักขระพิเศษ `\xa0` (Non-breaking Space) ที่ผสมอยู่ในชื่อเดือน ทำให้เดือนเดียวกันถูกนับเป็น 16 unique แทนที่จะเป็น 12
- ตรวจสอบ Whitespace ในชื่อเขต (ไม่พบปัญหา) และตรวจสอบว่าทุกจุดข้อมูลมีค่า >= 0 (ไม่พบค่าติดลบ)
- สร้างคอลัมน์ `date` (datetime) โดยแปลง พ.ศ. → ค.ศ. (ลบ 543) เพื่อใช้เป็น Time Index สำหรับการวิเคราะห์อนุกรมเวลา
- สรุปผลการเตรียมข้อมูลเปรียบเทียบก่อน-หลังทำความสะอาด (Preparation Log) แสดง dtype, Missing, Unique ก่อน/หลังทุกคอลัมน์

### การวิเคราะห์

โจทย์ตั้งคำถามเชิงวิเคราะห์ 3 ข้อ :

**คำถามที่ 1 : การใช้ไฟฟ้ารวมทุกเขตมีแนวโน้มอย่างไร และให้ระบุจุดผิดปกติ (Trend & Anomaly Detection)**

- **วิธีการ** : ใช้ Moving Average 12 เดือน เป็นเส้น Trend เนื่องจาก Moving Average ช่วยให้ข้อมูลเรียบ ทำให้เห็นการเคลื่อนไหวแนวโน้มของข้อมูลชัดเจน นอกจากนี้เมื่อใช้ Bollinger Band (k คูณด้วย rolling STD.) จะทำให้ตรวจจับค่าที่อยู่นอกช่วง k * ส่วนเบี่ยงเบนมาตรฐานในอดีต โดยกำหนด k=2 (ประมาณ 95% ของข้อมูลในอดีต) จุดข้อมูลที่อยู่นอกแถบถูกจัดเป็น Anomaly พบ 3 จุด ได้แก่ Jan 2021 (โควิด 19 + ฤดูหนาว), May 2023 (ฟื้นเศรษฐกิจ + เอลนีโญ), Jan 2025 (กลับสู่สภาวะปกติหลังจากที่ปี 2024 ที่โตผิดปกติ)
    - Source Anomaly Detection ด้วย Bollinger Band : https://docs.datarobot.com/en/docs/modeling/special-workflows/unsupervised/anomaly-detection.html
- **Visualization 1.1** : Line Chart + Rolling Mean 12 เดือน + Bollinger Band (+-2 SD) แสดง Trend พร้อมจุด Anomaly
- **Visualization 1.2** : Box Plot รายเดือนเพื่อดูอิทธิพลของฤดูกาล พบว่าพีคเดือน พ.ค. ต่ำสุดเดือน ธ.ค. นอกจากนี้ยังคำนวณ **Seasonal Index** (ดัชนีฤดูกาล) ด้วยสูตร SI หรือ Seasonal Index(%) = (Average for the Season / Average of all data) x 100 เพื่อวัดว่าเดือนไหนมียอดใช้ไฟสูง/ต่ำกว่าค่าเฉลี่ยทั้งปีเท่าไหร่

**คำถามที่ 2 : เขตไหนใช้ไฟมากที่สุด และเขตไหนโตเร็วที่สุด? (District Ranking & Growth)**

- **วิธีการ** : คำนวณ **CAGR (Compound Annual Growth Rate)** ของแต่ละเขตเพื่อวัดอัตราการเติบโตแบบทบต้นว่าเขตไหนโตเร็วที่สุดในช่วง 7 ปี
- จัดอันดับเขตตามค่าเฉลี่ยการใช้ไฟฟ้าต่อเดือน (แกน X) และอัตราการเติบโตสะสม (แกน Y) แล้วใช้ค่ามัธยฐาน (Median) ของทั้งสองตัวแปรเป็นจุดตัดแบ่ง 4 กลุ่ม (Quadrant) บน Scatter Plot ทำให้ระบุได้ว่าเขตไหนใช้ไฟมาก+โตเร็ว และ เขตไหนใช้น้อยแต่โตเร็ว

**คำถามที่ 3 : รูปแบบการใช้ไฟฟ้าตามฤดูกาลต่างกันระหว่างเขตไหม? (Seasonal Pattern by District)**

- **วิธีการ** : คำนวณ **ดัชนีฤดูกาลรายเขต (Seasonal Index by District)** 
- **Visualization 3.1** : Heatmap แสดง Seasonal Index (เดือน x เขต) ที่ Normalize แล้ว ช่องสีแดงเข้ม = เดือน Peak สูงกว่าค่าเฉลี่ย, ช่องน้ำเงินเข้ม = ต่ำกว่า
- **Visualization 3.2** : Bar Chart จัดอันดับเขตตาม **SD ของ Seasonal Index** เพื่อจำแนกความผันผวนตามฤดูกาล เขตที่มี SD สูงสะท้อนว่าการใช้ไฟเปลี่ยนแปลงตามช่วงเวลามาก (มักเป็นย่านที่อยู่อาศัย) เขตที่มี SD ต่ำสะท้อนการใช้ไฟสม่ำเสมอตลอดทั้งปี (มักเป็นย่านอุตสาหกรรม/ธุรกิจ)

**การวิเคราะห์เพิ่มเติม : ความสัมพันธ์ระหว่างอุณหภูมิกับปริมาณการใช้ไฟฟ้า**

- **Visualization 4.1** : Stacked Area Chart แสดงสัดส่วน (Share) แต่ละเขตเทียบกับยอดรวม overtime
- **Visualization 4.2** : Scatter Plot + OLS Trendline (Ordinary Least Squares) แสดงทิศทางความสัมพันธ์ระหว่างอุณหภูมิกับการใช้ไฟฟ้า
- ไปดึงข้อมูลอุณหภูมิรายเดือนกรุงเทพจากศูนย์ข้อมูลเปิดภาครัฐ (Source : https://data.go.th/dataset/tmax-tmin) แล้ว preprocess มาเป็นข้อมูลอุณหภูมิเฉลี่ยรายเดือนเฉพาะกรุงเทพ
- ใช้ **Pearson Correlation** คำนวณค่าสหสัมพันธ์และ p-value ระหว่างอุณหภูมิเฉลี่ยรายเดือนกับปริมาณการใช้ไฟฟ้ารวม ตรวจสอบนัยสำคัญทางสถิติโดยใช้เกณฑ์มาตรฐานที่ระดับ 0.05 พบว่า r = 0.8 และ p-value เข้าใกล้ 0 สรุปว่ามีความสัมพันธ์ในทางเดียวกันอย่างมีนัยสำคัญทางสถิติในระดับสูงมาก


### สรุป Findings & Actions

- ควร stock ไฟฟ้าสำรองช่วงเดือน พ.ค. (Peak) และรัฐควรออกมาตรการค่าไฟช่วงฤดูร้อน (มี.ค.-พ.ค.)
- เขตที่ใช้ไฟมากและยังโตต่อ (บางกะปิ, บางพลี, บางเขน, บางนา) ภาครัฐควรลงทุนโครงสร้างพื้นฐาน เขตที่โตเร็ว (มีนบุรี, บางใหญ่, ลาดกระบัง) ภาคเอกชนอาจลงทุนอสังหาริมทรัพย์รองรับ
- เขตที่ผันผวนสูงควรสำรองพลังงานเป็นพิเศษ ควรโฆษณาลดการใช้ไฟในฤดูร้อนสำหรับย่านที่อยู่อาศัย
- ใช้ผลพยากรณ์อากาศมาพยากรณ์ปริมาณการใช้ไฟฟ้าเพื่อวางแผนนโยบายด้านพลังงานและสำรองงบประมาณ


---

## ข้อ 2 : Mini Hackathon 2 - OCR (สกัดผลเลือกตั้งจากภาพเอกสาร)

### โจทย์

สกัดข้อมูลคะแนนเลือกตั้ง สส. จากภาพถ่ายเอกสารผลการเลือกตั้งทั้งแบบแบ่งเขต และแบบบัญชีรายชื่อ ให้ได้ไฟล์ CSV ตาม Template ที่กำหนด 

### โมเดลที่ใช้

ใช้โมเดล **Gemini 3 Flash** เป็นตัว OCR หลัก เนื่องจากจากการใช้ Gemini 3 Flash มา OCR ในชีวิตประจำวัน คาดว่าตัวโมเดลมีประสิทธิภาพในการรับมือกับปัญหาที่พบจากการ EDA ในระดับหนึ่ง

### Preprocessing (EDA คุณภาพภาพ)

ก่อนจะส่งรูปเข้าโมเดล อ่านภาพทั้งหมด 846 ภาพ เพื่อดึงค่าเมตริกซ์ 4 ด้าน แสดงผลเป็น Distribution ของค่าสถิติที่วัดได้:

- **DPI & Size** : ดึง DPI จาก EXIF Metadata และตรวจสอบขนาด Width x Height ว่าไม่เล็กเกินไป ไม่พบภาพที่มีปัญหา (ทุกภาพประมาณ2480x3507 พิกเซล)
- **Blur (Laplacian Variance)** : คำนวณ Variance ของ Laplacian Filter กำหนด Threshold ไว้ที่ 100 ภาพที่มีค่าต่ำกว่านี้ถือว่าเบลอ พบภาพเบลอเพียง 4 ภาพ (0.5%) สาเหตุมาจากจุดดำจากการถ่ายเอกสาร
- **Darkness (Mean Intensity)** : คำนวณค่าเฉลี่ยความสว่างของภาพ Grayscale พิจารณาเพียงแค่ภาพมืดเกินไป เนื่องจากเอกสารมีพื้นหลังสีขาว ไม่พบภาพที่มีปัญหา
- **Noise Level** : หาค่า Absolute Difference ระหว่างภาพต้นฉบับกับภาพที่นำไปกัด Noise ด้วย Median Blur แล้วนำค่าที่ได้ทุกรูปมาหาค่าเฉลี่ย ไม่มีภาพใดที่มีสิ่งรบกวนมากกว่าที่กำหนดไว้

<u>**Note**</u> ผู้ทำขาดความรู้ด้าน image preprocessing จึงขอข้ามการปรับปรุงคุณภาพข้อมูลรูปภาพเหล่านี้ เพื่อให้สามารถ submit ทันเวลา และจาก EDA ข้างต้น คุณภาพภาพส่วนใหญ่อยู่ในเกณฑ์ดี

### Processing (OCR Pipeline)

สร้าง Pipeline แยกเป็น Module ใน Python script ต่างหาก :

**Pipeline สำหรับเอกสารแบบแบ่งเขต :**
- กรองชื่อเอกสาร (doc_id) จากชื่อไฟล์รูปภาพตาม `submission_template_v3.csv` แล้วจัดกลุ่มหลายหน้าของเอกสารเดียวกัน
- ส่งภาพทุกหน้าของเอกสารเดียวกันไปให้ Gemini 3 Flash ประมวลผล โดยจะส่งเข้าไปหลายๆรูป (หลายหน้าของเอกสาร) แต่จะส่งทีละเอกสาร
- **Prompt Design (Text-First Approach)** : ออกแบบ Prompt ให้ยึดการอ่าน "คำอ่านคะแนน" (ตัวหนังสือภาษาไทย) เป็นหลักแล้วแปลงเป็นเลขอารบิก เนื่องจากมีสมมติฐานว่าข้อความภาษาไทยเชื่อถือได้มากกว่าตัวเลขไทย มี Fallback Rule (ถ้าอ่านคำไทยไม่ได้ ให้ใช้ตัวเลข) และ Multi-page Handling (รวมข้อมูลจากหลายหน้า)
- output จาก Gemini จะส่งผลลัพธ์ออกมาเป็น Markdown Table เก็บไว้ในไฟล์ `ชื่อเอกสาร (doc_id).md` ลงในโฟลเดอร์ `output/` โดยอัตโนมัติ

**Pipeline สำหรับเอกสารแบบบัญชีรายชื่อ :**
- ใช้ Pipeline แบบเดียวกัน แต่ปรับ Prompt ให้เหมาะกับรูปแบบเอกสารบัญชีรายชื่อ (โครงสร้างตารางต่างจากแบบแบ่งเขต)

### Post-processing (แปลง Markdown → CSV)

- เมื่อสแกนครบทั้งผลรายเขตและบัญชีรายชื่อแล้ว นำผลจากไฟล์ `.md` เหล่านั้นมา Map คะแนนโหวตกับชื่อพรรคให้ตรงกัน
- ใช้ **Fuzzy Matching** สำหรับชื่อพรรคที่ OCR อาจอ่านได้ไม่ตรง 100% (เช่น สะกดต่างกันเล็กน้อย) พร้อมแสดง Log ตรวจสอบความถูกต้องของการ Map ทุก doc_id
- Drop คอลัมน์ชื่อพรรค แล้ว Export เป็น CSV ตาม Template

### ผลลัพธ์การแข่งขัน

Kaggle Public Score : 0.2451 อันดับ อันดับ 69  || Private Score : 0.1747 อันดับ 84

---

## ข้อ 3 : Mini Hackathon 3 - FahMai (ระบบ RAG ถาม-ตอบสินค้า)

### โจทย์

สร้างระบบ Retrieval-Augmented Generation (RAG) สำหรับร้าน "ฟ้าใหม่ Electronics" เพื่อตอบคำถาม Multiple Choice 100 ข้อ (ตัวเลือก 1-10) เกี่ยวกับสินค้าอิเล็กทรอนิกส์ โดยใช้ฐานความรู้ที่เป็นไฟล์ Markdown (118 เอกสาร ได้แก่ ข้อมูลสินค้า 110 ไฟล์, นโยบาย 5 ไฟล์, ข้อมูลร้าน 3 ไฟล์)


### การทดลอง

**1. การเลือกโมเดล Embedding สำหรับการเข้า Encode ข้อความ**

โมเดลที่นำมาใช้ทดสอบ (Candidates) โดยกำหนดการทดลองให้มีความแตกต่างคือใช้ โมเดลใหญ่ และ โมเดลเล็ก :
- **bge-m3** (ตัวแทนโมเดลเล็ก)
    สาเหตุที่เลือกมา พบว่าได้รับการแนะนำจาก Community และมีผลการ Benchmark ที่ดีสำหรับงาน Retrieve (จากการทดสอบ Thai retrieval benchmark)
    - Source: https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark
- **Qwen3-Embedding-4B** (ตัวแทนโมเดลใหญ่)
    สาเหตุที่เลือกมา พบว่าผลการ Benchmark Text Embedding ภาษาไทยเทียบกับโมเดลหลายๆขนาดหลายๆรุ่น เป็นโมเดลขนาดใหญ่ (4B) ที่ได้ผลลัพธ์การประเมินที่ดีสุด
    - Source: https://github.com/anusoft/thai-mteb-leaderboard
- เปรียบเทียบคะแนนการทำข้อสอบโดยกำหนดโมเดล KBTG และ Prompt เดียวกันพบว่าโมเดล bge-m3 จะได้คะแนนสอบในช่วง **80-85%** ในขณะที่ โมเดล Qwen3-4B ได้รับคะแนนอยู่ในช่วง **90-93%** สำหรับ public score ดังนั้นในการสร้างระบบ RAG จะใช้โมเดล Qwen3-4B เป็นโมเดล Embedding

**กลยุทธ์การ Chunking** ทำการ Tuning (ทดลองเปลี่ยนค่าไปเรื่อยๆ) เพื่อหาค่า :
- Chunk Size : ทดลอง [256, 512, 1024, 2048] → พบว่า 1024 ดีที่สุด
- Overlap : จากการศึกษาของ Nvidia ควรใช้ค่าที่ 10-20% อย่างไรก็ตามพบว่าการตั้งค่าที่ 30% โมเดลจะตอบคำถามได้คะแนนสูงสุด
    - Source: https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses

### Retrieval & Generation

**2. การ Retrieval เพื่อหาวิธีที่จะทำให้การ Retrieval มีประสิทธิภาพสูงที่สุด**

- วิธีการ Retrieve : ทดลองทั้งวิธี Dense, BM25 และ Hybrid → Benchmark ด้วยคะแนนสอบ ผลการศึกษาพบว่าวิธี Hybrid แบบกำหนดน้ำหนัก 0.7 ให้กับ Dense และ 0.3 ให้กับ BM25 มักจะได้คะแนนสอบสูงที่สุด
    - **Dense Retrieval** : Encode ทุก Chunk เป็น Embedding ด้วย Qwen3-4B แล้วใช้ Cosine Similarity หา Chunk ที่ใกล้เคียงคำถามที่สุด
    - **Sparse Retrieval / BM25** : ตัดคำแล้วสร้าง BM25 Index สำหรับค้นหาแบบคำตรง (Term Matching)
    - **Hybrid Retrieval / RRF** : รวมผลจาก Dense (น้ำหนัก 0.7) และ BM25 (น้ำหนัก 0.3) เพื่อถ่วงดุลผลจากทั้ง 2 วิธีการ Search
- วิธีการ Reranking :
    - สำหรับขั้นตอนการประเมินผล Embedding ระหว่าง bge-m3 และ Qwen3-4B จะมีการใช้ Reranker ของโมเดลร่วมด้วยคือ
    - โมเดล bge-m3 จะใช้โมเดล bge-reranker-v2-m3 ร่วมด้วย (คะแนน 80%-85%)
    - โมเดล Qwen3-4B จะใช้โมเดล Qwen3-Reranker-4B ร่วมด้วย (คะแนน 90%-93%)
    - นำ Chunk 100 อันดับแรกจาก Hybrid มาให้ Reranker จัดอันดับใหม่ เลือก Top-10 เพื่อกรองเอกสารที่ตรงกับคำถามจริงๆ
- **ผลการทดลอง** ใช้โมเดล Qwen3-4B ร่วมกับ Reranker Qwen3-Reranker-4B ในการ Retrieve

**3. โมเดลสำหรับการสร้างข้อความ**

โมเดลที่นำมาใช้ทดสอบประสิทธิภาพการ Generate และตอบคำถาม ได้แก่ :
- OpenThaiGPT-ThaiLLM-8B-instruct-v7.2
- Pathumma-ThaiLLM-qwen3-8b-think-3.0.0
- Typhoon-S-ThaiLLM-8B-Instruct
- THaLLE-0.2-ThaiLLM-8b-fa

**โมเดล LLM สำหรับ Generate ที่นำมาใช้จริงโดยใช้เกณฑ์** : ความสามารถในการสร้างข้อความตามคำสั่ง และ คะแนนจากการตอบคำถาม → **THaLLE-0.2-ThaiLLM-8b-fa**

### สรุปผลการทดลอง : วิธีที่ใช้ในการสร้าง RAG System สำหรับ Final Model

- Embedding Model : Qwen3-Embedding-4B
- Retrieve Method : Hybrid (Weight 0.7 Dense : 0.3 BM25)
- Reranker Model : Qwen3-Reranker-4B
- Generate Model : THaLLE-0.2-ThaiLLM-8b-fa

### เทคนิคเพิ่มเติม

- **Relevant Segment Extraction** : มีการศึกษาเทคนิคเพิ่มเติมสำหรับการ RAG จาก ***github.com/NirDiamant/RAG_Techniques*** และพิจารณาเลือกเทคนิค #11 Relevant Segment Extraction มาใช้ร่วมด้วย โดยรวม Chunk ที่อยู่ติดกันในเอกสารเดียวกันให้เป็นข้อความต่อเนื่อง ลดจำนวน Segment จาก 15 เหลือ 7 แต่ได้ข้อมูลครบถ้วนกว่า (Context ไม่ถูกตัดขาดกลางคัน)
    - Source: https://github.com/NirDiamant/RAG_Techniques

### Prompt Design & Generation

- ออกแบบ System Prompt ให้ LLM เป็น AI ผู้ช่วยร้านฟ้าใหม่ มีกฎการตอบชัดเจน (ตอบ 1-8 ถ้าพบคำตอบ, 9 ถ้าไม่พบข้อมูล, 10 ถ้าไม่เกี่ยวกับร้าน) แล้วส่งคำถาม ดำเนินการ Retrieval แล้วตอบคำถามโดย LLM


### การวัดผล

วัดผลด้วยคะแนนสอบจากการ submit บน Kaggle โดย Benchmark ทุกการทดลอง (เปลี่ยน Embedding, เปลี่ยน Retrieve Method, เปลี่ยน LLM) จากรอบการตอบที่ดีที่สุดพบว่าได้คะแนน

Kaggle Public Score : 0.93% อันดับ 114 || Private Score : 0.95 อันดับ 86

---

## โจทย์เสริม Mini Hackathon 4 (5 Domain)

- ทำแค่ 2 โจทย์จาก 5 โจทย์เนื่องจากข้อจำกัดด้านระยะเวลาที่กระชั้นชิดเนื่องจากการแข่งขันนี้แจ้งล่วงหน้าไม่นาน และทั้ง 2 โจทย์ที่ทำไม่มีโจทย์ใดที่ผ่าน baseline 

## ข้อ 4 : Mini Hackathon 4 - Sleep Stage Classification

### โจทย์

จำแนกระยะการนอนหลับ (Sleep Stage) 5 คลาส (Wake, N1, N2, N3, REM) จากข้อมูลสัญญาณชีวภาพ (Wearable Sensor) ที่เก็บจากอุปกรณ์สวมใส่ ประกอบด้วย ACC (X,Y,Z), BVP, EDA, TEMP โดยข้อมูล Train มี 83 คืน และ Test แบ่งเป็น Segment ย่อย

### โมเดลที่ใช้

ใช้โมเดล **U-Net (UNet)** ปรับจากงานวิจัย https://openreview.net/forum?id=ebRFWvAUjS สำหรับทำ Sequence-to-Sequence Classification โดยรับ Spectrogram และทำนายระยะการนอนทุก 30 วินาที 
### Preprocessing (Signal Selection & Normalization) (อ้างอิงจากงานวิจัย)

**1. การเลือกสัญญาณและปรับข้อมูล**
- **สัญญาณหลัก** : ACC (X,Y,Z), BVP, EDA, TEMP (ตัด HR, IBI ทิ้ง)
- **EDA (Phasic)** : สกัดเฉพาะส่วน Phasic ด้วย Highpass Butterworth Filter
- **TEMP (Delta)** : แปลงข้อมูลด้วยต่างลำดับที่หนึ่ง (First-order difference) 
- **Normalization** :
    - Median centering + Adaptive IQR (Sliding window 300s)
    - Outlier clipping : ตัดที่ 20x IQR (ACC, BVP, EDA) และ 15x IQR (TEMP)
    - Z-score normalization

**2. การปรับความถี่และการแปลง Spectrogram**
- **Resampling** : จัดการข้อมูลใหม่จาก 16Hz เป็น **32Hz**
- **Feature Extraction** : ใช้ **STFT (Short-Time Fourier Transform)** (n_fft=256, hop_length=64) แปลงสัญญาณ 1D เป็น Spectrogram 2D เพื่อให้โมเดล 2D CNN สกัด Feature จาก Time-Frequency Representation ได้

**3. การจัดโครงสร้างข้อมูล (Input Structuring)**
- **Epoching** : 30 วินาที/Epoch
- **Segment Size** : รวม 1,024 Epochs รวดเดียวต่อ 1 ตัวอย่าง (Seq-to-Seq)

### โครงสร้างโมเดลและ Training

**4. โครงสร้างโมเดล (2D U-Net Architecture)**
- **M = 10** : จำนวนจุด Encoder / Decoder
- **K = 16** : Kernel height สำหรับ 2D Convolutions
- **Initial Filters = 32** : เริ่มต้นที่ 32 ตั้ง MAX ที่ 256


**5. Training**
- ใช้ **Focal Loss** (α=0.15, γ=2.0) เพื่อแก้ปัญหา Class Imbalance เนื่องจากระยะ N2 มีจำนวนมากกว่า N1/N3 อย่างมาก หลักการคือลดน้ำหนักของตัวอย่างที่จำแนกถูกแล้ว และเพิ่มน้ำหนักให้ตัวอย่างที่ยาก
- ใช้ **GroupKFold 5-Fold** (แบ่งตาม Subject) เพื่อป้องกันไม่ให้ข้อมูลของคนเดียวกันปรากฏทั้งใน Train และ Validation (ป้องกัน Data Leakage)
- Adam Optimizer (lr=1e-3), Early Stopping (Patience=5), Mixed Precision (FP16)
- มีการทำ Grid Search Hyperparameters เพื่อหา Config ที่ดีที่สุด

**6. Final Training**
- เทรนด้วยข้อมูลทั้งหมด 50 Epochs ด้วย Config ที่ดีที่สุดจาก Grid Search

### Inference

- ประมวลผล Test set โดย Preprocess เหมือน Train แล้วส่งเข้าโมเดล
- แปลงผลกลับเป็น Label (W/N1/N2/N3/R) แล้ว Export เป็น csv

### การวัดผล

วัดผลด้วย **Weighted F1 Score** 

Kaggle Public Score : 0.36527 อันดับที่ 279 || Private Score : 0.37412 อันดับที่ 280

---

## ข้อ 5 : Mini Hackathon 4: Word Segmentation

### โจทย์

ตัดคำภาษาไทย (Thai Word Segmentation) จากข้อความ โดยจำแนกแต่ละตัวอักษรเป็น 3 คลาส: B_WORD (ต้นคำ), I_WORD (กลางคำ), E_WORD (โดยใช้ชุดข้อมูล LST20 Corpus สำหรับ Train/Validation และ Test

### โมเดลที่ใช้

ใช้โมเดล **HoogBERTa** (lst-nectec/HoogBERTa) โดยทำ Fine-Tuning เพิ่ม Token Classification Head


### Preprocessing (เตรียมข้อมูล)

**ชุดข้อมูล** : LST20 Corpus จาก NECTEC ประกอบด้วยข้อความที่ผ่านการตัดคำและ Annotate 

**การแปลงเป็น Character-level** : แปลงข้อมูลจากระดับคำเป็นระดับตัวอักษร โดยกำหนด :
- คำที่มี 1 ตัวอักษร → B_WORD
- คำที่มีหลายตัวอักษร → ตัวแรก: B_WORD, ตัวกลาง: I_WORD, ตัวสุดท้าย: E_WORD
- ช่องว่าง/whitespace → O (เพื่อให้โมเดลเข้าใจลักษณะของ Whitespace)


### Processing (Fine-Tuning)

โหลด HoogBERTa + Token Classification Head (4 Labels) แล้ว Train ด้วย HuggingFace Trainer :
- Learning Rate: 3e-5, Batch Size: 32, Epochs: 6
- Weight Decay: 0.01, Warmup Ratio: 0.1, FP16
- Evaluation Strategy: ทุก Epoch, Metric: Macro F1, Load Best Model at End
- Source : https://arxiv.org/pdf/2101.09635 (พารามิเตอร์)

### Inference

อ่านข้อความส่งเข้าโมเดลทีละช่วง  ตอนเลือกคำตอบสุดท้ายจะตัด Label O ออก (ใช้เฉพาะ B/I/E) แล้ว Export เป็น csv

### การวัดผล

วัดผลด้วย **Macro F1** 

Kaggle Public Score : 0.97012 อันดับที่ 80 (**ไม่ผ่าน baseline** 0.97153 อันดับที่ 71-72) || Private Score : core : 0.96978 อันดับที่ 81 (**ไม่ผ่าน baseline** 0.97174
 อันดับที่ 71-72)
