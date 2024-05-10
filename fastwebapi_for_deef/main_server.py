# 필요한 라이브러리들
import uvicorn
from fastapi import FastAPI, Request, File, UploadFile, Form, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, text , Column , String ,Integer
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import OperationalError
import shutil
from typing_extensions import Annotated
from sqlalchemy.ext.declarative import declarative_base
from typing import Optional
import os 
import subprocess

# MySQL 데이터베이스 연결 설정
DATABASE_URL = 'mysql+pymysql://{user}:{server_pass}@{yourip}:{yourport}/test'
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# FastAPI 인스턴스 생성
app = FastAPI()
DIRECTORY = '/fastwebapi_for_deef/'

Base = declarative_base()

class Video(Base):
    __tablename__ = "videos"

    vid_name = Column(String(100))
    vid_url = Column(String(100),primary_key=True, nullable=False)
    vid_fake = Column(String(20))
    vid_tag = Column(String(100))
     
def save_video_url(db: Session, vid_name: str, vid_url: str, vid_fake: str, vid_tag: str):
    db_video = Video(vid_name=vid_name, vid_url=vid_url, vid_fake=vid_fake, vid_tag=vid_tag)
    db.add(db_video)
    db.commit()
    db.refresh(db_video)
       


app.mount('/static', StaticFiles(directory=DIRECTORY + 'static'), name='static')
templates = Jinja2Templates(directory=DIRECTORY + 'templates')

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get('/')
def test_get(request: Request):
    return templates.TemplateResponse('main_upload.html', {'request': request})


@app.post('/result_page')
async def create_upload_file(request: Request, video: UploadFile = File(...), url: Optional[str] = Form(None), db: Session = Depends(get_db)):
    # 새로운 폴더 생성
    directory_path = os.path.join(DIRECTORY, "static", video.filename[:-4])
    os.makedirs(directory_path, exist_ok=True)
    # 파일 저장 경로 설정
    file_location = os.path.join(directory_path, video.filename)
    # 파일 저장
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
        
    # 모델 작동 
    result = subprocess.run(['python', '/main_play.py', '--video_path', file_location],capture_output=True)
    
    
    print('result.stdout.strip()' , result.stdout.strip())
    save_video_url(db, video.filename[:-4], url,result.stdout.strip(), '')
    return templates.TemplateResponse('result_page.html', {'request': request, 'vid_name': video.filename[:-4]})


@app.get('/list')
def mysqltest_get(request: Request, db: Session = Depends(get_db)):
    result_db = db.execute(text('SELECT * FROM videos')).fetchall()
    
    result = [{'vid_name': data[0],'vid_url': data[1], 'vid_tag': data[-1]} for data in result_db]  
    return templates.TemplateResponse('list.html', {'request': request, 'result_table': result})

@app.post('/list')
def mysqltest_get(request: Request, db: Session = Depends(get_db)):
    result_db = db.execute(text('SELECT * FROM videos')).fetchall()
    
    result = [{'vid_name': data[0],'vid_url': data[1] , 'vid_tag': data[-1]} for data in result_db]  
    return templates.TemplateResponse('list.html', {'request': request, 'result_table': result})


@app.get('/detail')
def test_post(request: Request, vid_name: str, db: Session = Depends(get_db)):
    query = text("SELECT * FROM videos WHERE vid_name = :vid_name")
    result_db = db.execute(query, {"vid_name": vid_name}).fetchall()
    print(result_db)
    result = [{'vid_name': data[0], 'vid_url': data[1], 'vid_fake': data[2], 'vid_tag': data[-1]} for data in result_db]
    
    return templates.TemplateResponse('detail.html', {'request': request, 'result_table': result})


@app.post('/update')
def post_update(request: Request, vid_name: str, vid_tag: Annotated[str, Form()], db: Session = Depends(get_db)):
    select_query = text("SELECT vid_tag FROM videos WHERE vid_name = :vid_name")
    current_tags = db.execute(select_query, {"vid_name": vid_name}).fetchone()

    if current_tags:
        current_tags = current_tags[0]
        updated_tags = f"{current_tags}, {vid_tag}"
    else:
        updated_tags = vid_tag

    update_query = text("UPDATE videos SET vid_tag = :vid_tag WHERE vid_name = :vid_name")
    db.execute(update_query, {"vid_name": vid_name, "vid_tag": updated_tags})
    db.commit()
    
    select_query = text("SELECT * FROM videos WHERE vid_name = :vid_name")
    result_db = db.execute(select_query, {"vid_name": vid_name}).fetchall()
    result = [{'vid_name': row[0], 'vid_url': row[1], 'vid_fake': row[2], 'vid_tag': row[-1]} for row in result_db]
    
    return templates.TemplateResponse('detail.html', {'request': request, 'result_table': result})


class Post(Base):
    __tablename__ = "post"
    post_number=Column(Integer,primary_key=True, nullable=False)
    post_name= Column(String(20), nullable=False)
    writer = Column(String(10), nullable=False)
    writer_pass = Column(String(20))
    post_text = Column(String(100))
    
def save_post(db: Session, post_name: str, writer: str, writer_pass: str, post_text: str):
    new_post = Post(post_name=post_name, writer=writer, writer_pass=writer_pass, post_text=post_text) 
    db.add(new_post)
    db.commit()
    db.refresh(new_post)


 
from fastapi import HTTPException
from fastapi.responses import HTMLResponse


@app.get("/post", response_class=HTMLResponse)
def get_post_form(request: Request):
    # 기존 폼 페이지 또는 기타 페이지 반환
    return templates.TemplateResponse("post.html", {"request": request})

@app.post('/post')
async def create_post(request: Request, post_name: str = Form(...), writer: str = Form(...), writer_pass: str = Form(...), post_text: str = Form(...), db: Session = Depends(get_db)):
    await save_post(db, post_name, writer, writer_pass, post_text)  # save_post가 비동기 함수라고 가정
    return {"message": "Post created successfully!"}

@app.get("/post_result")
async def post_result(request: Request, post_name: str = Form(...), writer: str = Form(...), writer_pass: str = Form(...), post_text: str = Form(...), db: Session = Depends(get_db)):
    save_post(db, post_name, writer, writer_pass, post_text)
    print(post_name)
    return templates.TemplateResponse("post_result.html", {"request": request})


@app.post("/post_result")
async def post_result(request: Request, post_name: str = Form(...), writer: str = Form(...), writer_pass: str = Form(...), post_text: str = Form(...), db: Session = Depends(get_db)):
    save_post(db, post_name, writer, writer_pass, post_text)
    print(post_name)
    return templates.TemplateResponse("post_result.html", {"request": request})


@app.get('/post_list')
def post_list(request: Request, db: Session = Depends(get_db)):
    # 데이터베이스에서 모든 포스트를 가져옵니다.
    result_db = db.query(Post).all()
    # 가져온 포스트를 원하는 형식으로 가공합니다.
    result = [{'post_number': post.post_number, 'post_name': post.post_name, 'writer': post.writer, 'writer_pass': post.writer_pass, 'post_text': post.post_text} for post in result_db]  
    print(result)
    # 결과를 포스트 목록 템플릿에 전달하여 렌더링합니다.
    return templates.TemplateResponse('post_list.html', {'request': request, 'post_list': result})

@app.post('/post_list')
def post_list(request: Request, db: Session = Depends(get_db)):
    # 데이터베이스에서 모든 포스트를 가져옵니다.
    result_db = db.query(Post).all()
    # 가져온 포스트를 원하는 형식으로 가공합니다.
    result = [{'post_number': post.post_number, 'post_name': post.post_name, 'writer': post.writer, 'writer_pass': post.writer_pass, 'post_text': post.post_text} for post in result_db]  
    print(result)
    # 결과를 포스트 목록 템플릿에 전달하여 렌더링합니다.
    return templates.TemplateResponse('post_list.html', {'request': request, 'post_list': result})



@app.get('/post_detail')
def test_post(request: Request, post_number: int, db: Session = Depends(get_db)):
    # 게시글 번호를 사용하여 해당 게시글 조회
    post = db.query(Post).filter(Post.post_number == post_number).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return templates.TemplateResponse('post_detail.html', {'request': request, 'result_table': [post]})


if __name__ =='__main__':
    uvicorn.run(app, host='localhost', port=40001)
