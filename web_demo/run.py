# -*- coding: utf-8 -*-
# from app.main_dummy import app
# 파일 바꿔주기
from app.main import app

if __name__ == "__main__":
    app.run(port=5000, debug=True)
    
