import uvicorn
from typing_extensions import Annotated
from fastapi.staticfiles import StaticFiles
import torchvision.transforms as transforms
from PIL import Image
import torch
from fastapi import FastAPI, Request, File, UploadFile ,Form
from fastapi.templating import Jinja2Templates
import os
import shutil

from sqlalchemy import create_engine
db_connection = create_engine('mysql+pymysql://alpaco:1234@112.175.29.231:3306/test')

app = FastAPI()
directory='/fastwebapi_for_deef/'
app.mount('/static',StaticFiles(directory=directory+'static'),name='static')
templates=Jinja2Templates(directory=directory+'templates')

templates_directory=directory+'templates/'
@app.get('/')
def test_get(request:Request):
    return templates.TemplateResponse('post_text.html',{'request':request})

@app.post('/result_page')
async def create_upload_file(request:Request,video: UploadFile = File(...)):
    with open(f"/fastwebapi_for_deef/upload_data/{video.filename}", "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    #return {"filename": video.filename}
    # output = infer(file_save_folder+file.filename)
    #return templates.TemplateResponse("result_page.html", {'request': request})#,'result':output})

    return templates.TemplateResponse('result_page.html',{'request':request})



@app.post('/list')
def mysqltest(request:Request):
    query = db_connection.execute('select * from videos')
    result_db = query.fetchall()
        
    result=[]
    for data in result_db:
        temp = {'vid_name':data[0],'vid_tag':data[-1]}
        result.append(temp)
    print(result)
    return templates.TemplateResponse('list.html',{'request':request,'result_table':result})


@app.get('/list')
def mysqltest(request:Request):
    query = db_connection.execute('select * from videos')
    result_db = query.fetchall()
        
    result=[]
    for data in result_db:
        temp = {'vid_name':data[0],'vid_tag':data[-1]}
        result.append(temp)
    print(result)
    return templates.TemplateResponse('list.html',{'request':request,'result_table':result})


@app.get('/detail')
def test_post(request:Request,vid_name:str):
    print('vid',vid_name)
    query = db_connection.execute("select * from videos where vid_name = '{}'".format(vid_name))
    result_db = query.fetchall()
    
    result = []
    for data in result_db:
        temp = {'vid_name':data[0], 'vid_url' : data[1] , 'vid_fake' :data[2],'vid_tag' : data[-1]}
        result.append(temp)
    return templates.TemplateResponse('detail.html',{'request':request,'result_table':result})


from fastapi import Request, Form
from sqlalchemy import text

@app.post('/update')
def post_update(request: Request, vid_name: str,vid_tag:Annotated[str,Form()]):

    if vid_name:
        update_query = text("UPDATE videos SET vid_tag ='{}' WHERE  vid_name ='{}'".format(vid_tag,vid_name))
        db_connection.execute(update_query, {"vid_name": vid_name, "vid_tag": vid_tag})
    
    select_query = text("SELECT * FROM videos WHERE vid_name ='{}' ".format(vid_name))
    result_db = db_connection.execute(select_query, {"vid_name": vid_name}).fetchall()

    result = []
    for row in result_db:
        temp = {'vid_name': row[0], 'vid_url': row[1], 'vid_fake': row[-2], 'vid_tag': row[-1]}
        result.append(temp)

    return templates.TemplateResponse('detail.html', {'request': request, 'result_table': result})




if __name__ =='__main__':
    uvicorn.run(app,host='localhost',port = 40001)