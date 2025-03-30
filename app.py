from flask import Flask, request, jsonify
from analysis_v6 import ultimate_process, Config

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # 获取输入参数
        data = request.get_json()
        file_path = data.get("file_path")
        file_type = data.get("file_type")  # "txt", "csv", or "image"
        api_key = data.get("api_key")      # 你的大模型 API 密钥

        if not all([file_path, file_type, api_key]):
            return jsonify({"error": "Missing required parameters"}), 400
        
        # 包装路径为列表
        filelist_path = [file_path]

        # 创建分析对象
        manager = ultimate_process(filelist_path=filelist_path,
                                   api_key=api_key,
                                   output_dir=Config.OUTPUT_DIR)
        # 执行主流程
        result = manager.main()

        return jsonify({"summary": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
