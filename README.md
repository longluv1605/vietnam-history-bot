# Chatbot hỏi đáp môn lịch sử phổ thông

## Cài đặt môi trường

1. Cài đặt `anaconda`
2. Cài đặt môi trường bot:

    ```bash
    conda env create -f environment.yml
    ```

3. Cài đặt Tessaract-OCR theo hướng dẫn [tại đây](https://docs.coro.net/featured/agent/install-tesseract-windows/).

## Cách chạy

1. Chỉnh sửa file `.env`
2. Cài đặt database:
    - Cài đặt Knowledge-graph:

        ```bash
        python src/graph_db/setup_graph.py
        python src/graph_db/setup_vector_index.py
        ```

    - Cài đặt text vector stores:

        ```bash
        python src/vector_db/setup_vector_db.py
        ```

3. Chạy bot trên nên console:

    ```bash
    python src/chatbot.py
    ```
