# gpt-from-scratch
 character-level GPT language model implemented from scratch in PyTorch, based on Andrej Karpathy's 'Zero to Hero' series.
# مدل زبان GPT بر پایه معماری Transformer

این پروژه یک پیاده‌سازی ساده از یک مدل زبان GPT در سطح کاراکتر (character-level) است که از ابتدا با استفاده از PyTorch ساخته شده است. این کد بر اساس دوره آموزشی "Neural Networks: Zero to Hero" توسط Andrej Karpathy نوشته شده است.

## توضیحات

این مدل از معماری Transformer برای پیش‌بینی کاراکتر بعدی در یک متن استفاده می‌کند. فایل `input.txt` شامل مجموعه آثار شکسپیر است که مدل بر روی آن آموزش می‌بیند.

## ویژگی‌ها

-   پیاده‌سازی کامل مکانیزم **Self-Attention**.
-   استفاده از **Multi-Head Attention** برای تمرکز بر روی بخش‌های مختلف ورودی.
-   ساختار **Block-based** که از لایه‌های Attention و Feed-Forward تشکیل شده است.
-   استفاده از **Token and Positional Embeddings**.

## نحوه اجرا

برای اجرای این پروژه، مراحل زیر را دنبال کنید:

1.  **یک محیط مجازی بسازید و آن را فعال کنید:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2.  **وابستگی‌ها را نصب کنید:**
    یک فایل با نام `requirements.txt` بسازید و `torch` را در آن قرار دهید. سپس دستور زیر را اجرا کنید:
    ```bash
    pip install -r requirements.txt
    ```

3.  **اسکریپت را اجرا کنید:**
    ```bash
    python gpt_dev.py
    ```
    اسکریپت شروع به آموزش مدل می‌کند و در نهایت یک متن ۵۰۰ کاراکتری تولید خواهد کرد.

## قدردانی

این پروژه به شدت از آموزش‌های ویدیویی فوق‌العاده **Andrej Karpathy** الهام گرفته شده است.
[لینک به پلی‌لیست یوتیوب](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
