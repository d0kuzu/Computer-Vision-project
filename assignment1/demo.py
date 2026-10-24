"""Local-only drawing demo: python demo.py, then open http://127.0.0.1:8765."""
from http.server import BaseHTTPRequestHandler,HTTPServer
from io import BytesIO
from pathlib import Path
import base64,json
from PIL import Image,ImageOps
from predict import load_model,predict_image

PAGE='''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Распознавание рукописных цифр</title><style>
*{box-sizing:border-box}body{margin:0;background:#eef2f8;color:#172b44;font:17px system-ui}main{max-width:920px;margin:55px auto;padding:32px;background:white;border-radius:22px;box-shadow:0 18px 70px #152a4314}h1{font-size:32px;margin:12px 0}p{line-height:1.55;color:#53657a}.tag{color:#326bd2;font-size:12px;letter-spacing:2px;font-weight:700}.grid{display:flex;gap:40px;flex-wrap:wrap;margin-top:25px}canvas{border:2px solid #c8d5e6;border-radius:12px;touch-action:none;width:280px;height:280px;background:white}button{border:0;border-radius:8px;background:#235bcc;color:white;padding:12px 18px;font:inherit;cursor:pointer;margin:12px 8px 0 0}button.secondary{background:#e7edf6;color:#243c5b}#answer{font-size:80px;font-weight:750;line-height:1.2}.result{flex:1;min-width:240px}.bar{height:7px;background:#e8eef7;border-radius:4px;margin:5px 0 12px}.fill{height:7px;background:#326bd2;border-radius:4px}.note{font-size:13px;border-top:1px solid #e0e7f0;padding-top:20px;margin-top:28px}</style>
<main><span class="tag">COMPUTER VISION · MINI PROJECT</span><h1>Распознавание рукописных цифр</h1><p>Нарисуйте одну цифру от 0 до 9. Нейросеть покажет предсказание и три наибольших значения softmax.</p><div class="grid"><div><canvas id="pad" width="280" height="280" aria-label="Поле рисования цифры"></canvas><br><button id="go">Распознать</button><button id="clear" class="secondary">Очистить</button></div><div class="result"><p id="status" role="status">Жду рисунок</p><div id="answer">—</div><div id="scores"></div></div></div><p class="note">Модель обучена на открытом наборе MNIST. Это демонстрация для цифр, не для математических знаков. Значения softmax не являются проверенной вероятностью правильного ответа. Рисунок обрабатывается локально и не сохраняется.</p></main><script>
const canvas=document.getElementById('pad'),ctx=canvas.getContext('2d'),status=document.getElementById('status'),answer=document.getElementById('answer'),scores=document.getElementById('scores');let drawing=false,hasInk=false;
function clear(){ctx.fillStyle='white';ctx.fillRect(0,0,280,280);hasInk=false;answer.textContent='—';scores.replaceChildren();status.textContent='Жду рисунок'}
function pos(e){const r=canvas.getBoundingClientRect();return [(e.clientX-r.left)*280/r.width,(e.clientY-r.top)*280/r.height]}
canvas.onpointerdown=e=>{drawing=true;hasInk=true;canvas.setPointerCapture(e.pointerId);ctx.strokeStyle='black';ctx.lineWidth=17;ctx.lineCap='round';ctx.lineJoin='round';ctx.beginPath();ctx.moveTo(...pos(e));ctx.lineTo(pos(e)[0]+.1,pos(e)[1]+.1);ctx.stroke()};canvas.onpointermove=e=>{if(drawing){ctx.lineTo(...pos(e));ctx.stroke()}};canvas.onpointerup=canvas.onpointercancel=()=>drawing=false;
document.getElementById('clear').onclick=clear;document.getElementById('go').onclick=async()=>{if(!hasInk){status.textContent='Сначала нарисуйте цифру';return}status.textContent='Распознаю…';try{const res=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:canvas.toDataURL('image/png')})});const data=await res.json();if(!res.ok)throw new Error(data.error);answer.textContent=data.digit;status.textContent='Предсказание CNN';scores.replaceChildren();for(const item of data.scores.slice(0,3)){const row=document.createElement('div');row.textContent=item.digit+' — '+(item.score*100).toFixed(1)+'%';const bar=document.createElement('div');bar.className='bar';const fill=document.createElement('div');fill.className='fill';fill.style.width=(item.score*100)+'%';bar.append(fill);scores.append(row,bar)}}catch(err){status.textContent='Ошибка: '+err.message}};clear();</script></html>'''

class Handler(BaseHTTPRequestHandler):
    def respond(self,status,payload,kind='application/json; charset=utf-8'):
        raw=payload.encode('utf-8'); self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    def do_GET(self):
        if self.path=='/': self.respond(200,PAGE,'text/html; charset=utf-8')
        else:self.respond(404,json.dumps({'error':'Not found'}))
    def do_POST(self):
        if self.path!='/predict':self.respond(404,'{}');return
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<500000:raise ValueError('Invalid image size')
            payload=json.loads(self.rfile.read(length))
            encoded=payload['image'].split(',',1)[1]
            with Image.open(BytesIO(base64.b64decode(encoded,validate=True))) as raw:
                if raw.size != (280,280):raise ValueError('Expected a 280 x 280 drawing')
                image=raw.convert('L')
            ink=ImageOps.invert(image); bbox=ink.point(lambda p:255 if p>35 else 0).getbbox()
            if bbox is None:raise ValueError('Draw a digit first')
            # Keep character position stable; add MNIST-like margin before shared preprocessing.
            crop=image.crop(bbox); side=max(crop.size); margin=max(3,int(side*.18))
            square=Image.new('L',(side+2*margin,side+2*margin),255)
            square.paste(crop,((square.width-crop.width)//2,(square.height-crop.height)//2))
            result=predict_image(self.server.model,self.server.classes,square)
            self.respond(200,json.dumps(result))
        except (ValueError,KeyError,OSError) as exc:self.respond(400,json.dumps({'error':str(exc)}))

if __name__=='__main__':
    server=HTTPServer(('127.0.0.1',8765),Handler)
    server.model,server.classes=load_model()
    print('Open http://127.0.0.1:8765 ; Ctrl+C to stop.',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
