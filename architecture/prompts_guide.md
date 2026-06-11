# مستند نصوص التوليد
## Prompts Repository

```
invokeai/
├── docs/
│   └── architecture/
│       └── prompts_guide.md  <-- هذا الملف
```

---

## مقدمة

هذا المستند يحتوي على مجموعة من النصوص التوليدية (Prompts) الجاهزة والمتوافقة مع:
- **نموذج:** Stable Diffusion 1.5
- **أبعاد:** 512x512
- **جهاز المعالجة:** CPU
- **أسلوب التصميم:** Quiet Luxury / Minimalist

---

## ملاحظات مهمة قبل الاستخدام

### إعدادات مُثلى لـ CPU:
| المعلمة | القيمة الموصى بها |
|---|---|
| **Steps** | 20-30 (الأفضل: 25) |
| **CFG Scale** | 7-8 (الأفضل: 7.5) |
| **Scheduler** | DPMSolverMultistep أو Euler a |
| **Seed** | ثابت للمعالجة المتكررة |
| **Size** | 512x512 |
| **Clip Skip** | 2 |

### نصائح عامة:
1. **ابدأ بسطر وصفى واحد** ثم أضف التفاصيل تدريجياً
2. **استخدم القوسين المزدوجين** `()` لتعزيز الكلمة بنسبة 1.1x
3. **استخدم الأقواس** `[]` لإنقاص الكلمة
4. **افصل الأفكار بفاصلة** `,`
5. **استخدم Negative Prompt** لاستبعاد العناصر غير المرغوبة

---

## 1. مشاهد داخلية - فخامة هادئة (Interior Design)

### 1.1. غرفة معيشة فاخرة
```
Positive: (minimalist luxury living room:1.3), (neutral tones:1.2), (clean lines:1.1),
floor-to-ceiling windows, natural light, (high-end furniture:1.1), marble coffee table,
(soft shadows:1.0), warm ambient lighting, (elegant simplicity:1.2), interior design magazine,
8k, ultra detailed, photorealistic

Negative: ugly, blurry, low quality, distorted, deformed, text, watermark,
busy patterns, cluttered, messy, cartoon, anime
```

### 1.2. مطبخ عصري
```
Positive: (modern minimalist kitchen:1.3), (white marble countertops:1.2),
(stainless steel appliances:1.1), (clean aesthetic:1.2), natural light streaming in,
(open shelving:1.0), (simple elegance:1.1), architectural photography,
warm wood accents, soft lighting, 8k, ultra realistic

Negative: dirty, cluttered, old, rusty, dark, low quality, blurry,
cartoon, anime, text, watermark, noise
```

### 1.3. حمام فندقي
```
Positive: (luxury hotel bathroom:1.3), (freestanding bathtub:1.2),
(double vanity:1.1), (marble tiles:1.1), (spa-like atmosphere:1.2),
warm lighting, (clean lines:1.1), elegant fixtures, soft towels,
minimalist design, photorealistic, 8k

Negative: dirty, old, stained, dark, low quality, blurry,
cartoon, noise, watermark, text
```

---

## 2. مشاهد خارجية - طبيعة هادئة (Landscape)

### 2.1. شاطئ هادئ
```
Positive: (serene beach at golden hour:1.3), (minimalist composition:1.2),
soft waves, warm sand, (gentle sunlight:1.1), (calm atmosphere:1.2),
clean horizon, (pastel sky:1.1), peaceful scene, nature photography,
8k, ultra detailed, cinematic

Negative: stormy, dark, ugly, blurry, low quality, crowded,
people, buildings, text, watermark, noise
```

### 2.2. غابة ضبابية
```
Positive: (misty forest morning:1.3), (soft diffused light:1.2),
(dense trees:1.1), (foggy atmosphere:1.2), (minimalist nature:1.1),
serene, tranquil, (natural beauty:1.1), ethereal mood,
landscape photography, 8k, ultra realistic

Negative: bright, sunny, clear, ugly, blurry, low quality,
text, watermark, people, buildings, noise
```

### 2.3. جبال مغطاة بالثلوج
```
Positive: (snow-covered mountains:1.3), (pristine white landscape:1.2),
(clear blue sky:1.1), (minimalist winter scene:1.2), (crisp details:1.1),
serene, peaceful, (natural beauty:1.1), (alpine landscape:1.0),
photorealistic, 8k, ultra detailed

Negative: dirty, muddy, ugly, blurry, low quality, summer,
text, watermark, people, noise
```

---

## 3. منتجات فاخرة (Luxury Products)

### 3.1. ساعة يدوية
```
Positive: (luxury watch on marble surface:1.3), (minimalist product photography:1.2),
(soft studio lighting:1.1), (clean background:1.2), (elegant design:1.1),
(sharp focus:1.0), (high-end brand aesthetic:1.1), professional product shot,
8k, ultra detailed, photorealistic

Negative: blurry, low quality, cheap, dirty, scratched,
text, watermark, cartoon, noise
```

### 3.2. عطر فاخر
```
Positive: (luxury perfume bottle:1.3), (minimalist product shot:1.2),
(soft gradient background:1.1), (elegant packaging:1.1),
(professional lighting:1.2), (clean composition:1.1),
high-end beauty photography, 8k, ultra detailed, photorealistic

Negative: blurry, low quality, cheap, dirty, broken,
text, watermark, cartoon, noise
```

### 3.3. حقيبة يد
```
Positive: (designer handbag:1.3), (minimalist product photography:1.2),
(leather texture detail:1.1), (soft lighting:1.1), (clean background:1.2),
luxury fashion, (elegant simplicity:1.1), professional product shot,
8k, ultra realistic

Negative: blurry, low quality, cheap, worn, dirty,
text, watermark, cartoon, noise
```

---

## 4. مشاهد معمارية (Architecture)

### 4.1. فيلا عصرية
```
Positive: (modern luxury villa:1.3), (clean architectural lines:1.2),
(large glass windows:1.1), (minimalist facade:1.2), (landscaped garden:1.1),
(geometric shapes:1.1), (natural materials:1.0), architectural photography,
golden hour lighting, 8k, ultra detailed, photorealistic

Negative: old, dilapidated, ugly, blurry, low quality, cluttered,
text, watermark, noise, cartoon
```

### 4.2. مكتب منزلي
```
Positive: (minimalist home office:1.3), (clean desk setup:1.2),
(large windows:1.1), (natural light:1.2), (ergonomic furniture:1.1),
(organized workspace:1.2), (modern design:1.1), interior design,
soft shadows, 8k, ultra realistic

Negative: messy, cluttered, dark, ugly, blurry, low quality,
text, watermark, noise, cartoon
```

---

## 5. مشاهد طعام (Food & Beverage)

### 5.1. قهوة فاخرة
```
Positive: (artisan coffee on marble countertop:1.3), (minimalist food photography:1.2),
(soft morning light:1.1), (steam rising:1.0), (clean composition:1.2),
(elegant cup:1.1), (warm tones:1.1), professional food shot,
8k, ultra detailed, photorealistic

Negative: blurry, low quality, dirty, spilled, ugly,
text, watermark, noise, cartoon
```

### 5.2. حلويات أنيقة
```
Positive: (elegant French pastry:1.3), (minimalist plating:1.2),
(soft studio lighting:1.1), (clean white plate:1.1),
(professional food photography:1.2), (simple elegance:1.1),
8k, ultra detailed, photorealistic

Negative: messy, ugly, low quality, blurry, burnt,
text, watermark, noise, cartoon
```

---

## 6. مشاهد أزياء (Fashion)

### 6.1. إطلالة أنيقة
```
Positive: (minimalist fashion editorial:1.3), (neutral color palette:1.2),
(soft natural light:1.1), (clean background:1.2), (elegant pose:1.1),
(luxury fabric texture:1.0), (high fashion aesthetic:1.1),
fashion photography, 8k, ultra detailed, photorealistic

Negative: ugly, blurry, low quality, distorted, deformed,
text, watermark, noise, cartoon
```

---

## 7. نصوص Negative Prompt通用 (Universal Negative)

### 7.1. للصور الفوتوغرافية
```
Negative: ugly, blurry, low quality, distorted, deformed, disfigured,
bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs,
mutated hands, extra fingers, missing fingers, bad hands, fused fingers,
too many fingers, long neck, text, watermark, signature, logo, username
```

### 7.2. للصور الرسمية
```
Negative: blurry, low quality, noise, grain, artifacts, jpeg artifacts,
overexposed, underexposed, oversaturated, desaturated, text, watermark,
signature, logo, copyright
```

### 7.3. للصور الجسيمة
```
Negative: ugly, creepy, scary, horror, dark, gloomy, blurry, low quality,
distorted, deformed, text, watermark, noise, cartoon, anime
```

---

## 8. جدول المعلمات المقترحة

| نوع المشهد | Steps | CFG | Scheduler | Seed |
|---|---|---|---|---|
| داخلي | 25 | 7.5 | DPM++ 2M Karras | ثابت |
| خارجي | 30 | 7.0 | Euler a | ثابت |
| منتج | 25 | 8.0 | DPM++ SDE Karras | ثابت |
| معماري | 30 | 7.5 | DPM++ 2M Karras | ثابت |
| طعام | 25 | 7.0 | Euler a | ثابت |
| أزياء | 30 | 7.5 | DPM++ 2M Karras | ثابت |

---

## 9. قواعد كتابة النصوص الفعّالة

### 9.1. البنية المقترحة
```
[النوع/النمط] + [الموضوع الرئيسي] + [التفاصيل] + [الإضاءة] + [الجودة]
```

### 9.2. الكلمات المُعزّزة للجودة
- `8k, ultra detailed, photorealistic`
- `professional photography`
- `studio lighting`
- `sharp focus`
- `high resolution`
- `award winning`

### 9.3. الكلمات المُعزّزة للأسلوب
- `minimalist`
- `clean lines`
- `elegant`
- `luxury`
- `high-end`
- `premium`

### 9.4. كلمات الإضاءة
- `natural light`
- `soft lighting`
- `golden hour`
- `studio lighting`
- `ambient lighting`
- `dramatic lighting`
