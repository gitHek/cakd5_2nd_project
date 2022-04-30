# -*- coding: utf-8 -*-
# from app.main_dummy import app
# 파일 바꿔주기
from app.main import app

if __name__ == "__main__":
    # 코랩용
    app.run()
    # 로컬용 
    # app.run(port=6006, debug=True)
