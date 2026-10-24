#!/usr/bin/env python3
"""Render recorded solver paths; CPU only, no model loading or route repair."""
import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tilebased_core_v85 import evaluate_path, validate_path, LAND_PROTOCOL


def font(size):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default()


def render(image, source, target, predicted, gt, arcs, title, note, side=480, region_model=None, land=False):
    rows, cols = image['shape']
    rgb = Image.fromarray(image['rgb']).resize((side, side), Image.Resampling.NEAREST)
    top, bottom, gap = 52, 44, 16
    panel_h = side + top + bottom
    canvas = Image.new('RGB', (side*2+gap*3, panel_h*2+gap*3+64), '#edf1f5')
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 10), title, font=font(19), fill='#172a3a')
    draw.text((gap, 37), note, font=font(14), fill='#34495e')
    def point(rc):
        r, c = rc
        return ((c+.5)*side/cols, (r+.5)*side/rows)
    def path(d, route, color, width=4):
        if not route:
            return
        pts = [point(p) for p in route]
        if len(pts)>1:
            d.line(pts, fill='white', width=width+3)
            d.line(pts, fill=color, width=width)
            # Direction indicators every few steps; never interpolate new routes.
            for i in range(0, len(pts)-1, max(1, (len(pts)-1)//8)):
                a, b = np.array(pts[i]), np.array(pts[i+1])
                v=b-a; length=np.linalg.norm(v)
                if length:
                    v=v/length; normal=np.array([-v[1],v[0]])
                    tip=a+(b-a)*.72
                    d.polygon([tuple(tip),tuple(tip-v*8+normal*4),tuple(tip-v*8-normal*4)], fill=color)
    panels = [
        ('1. Original input + public S/G', None, None, 'Top-down image; rows down, columns right'),
        ('2. Qwen model -> solver path', predicted, '#e21b63', 'Pink = recorded solver path; arrows = direction'),
        ('3. Land reference' if land else '3. Original dataset GT', gt, '#008fa8', 'Cyan = minimum-cost reference under current rules'),
        ('4. Proposed graph + solver path', predicted, '#e21b63', 'Orange = modeled areas; blue = predicted water' if region_model is not None else 'Orange = compiled connections; pink = chosen path'),
    ]
    for index,(label,route,color,caption) in enumerate(panels):
        x=gap+(index%2)*(side+gap); y=64+gap+(index//2)*(panel_h+gap)
        draw.rectangle((x,y,x+side,y+panel_h),fill='white')
        draw.text((x+10,y+14),label,font=font(18),fill='#172a3a')
        layer=rgb.copy(); d=ImageDraw.Draw(layer)
        if index==3 and region_model is not None:
            overlay=Image.new('RGBA',(side,side),(0,0,0,0)); od=ImageDraw.Draw(overlay)
            def draw_geometry(geom,fill,outline):
                if isinstance(geom,dict) and 'polygon_rc' in geom:
                    points=[(c*side/cols,r*side/rows) for r,c in geom['polygon_rc']]
                    od.polygon(points,fill=fill)
                    od.line(points+points[:1],fill=outline,width=2)
                else:
                    r0,c0,r1,c1=geom['bounds_rc'] if isinstance(geom,dict) else geom
                    od.rectangle((c0*side/cols,r0*side/rows,(c1+1)*side/cols-1,(r1+1)*side/rows-1),
                                 fill=fill,outline=outline,width=2)
            for region in region_model.get('regions',[]):
                draw_geometry(region,(255,170,0,25),(255,150,0,240))
            for geom in region_model.get('blocked_regions',[]):
                draw_geometry(geom,(0,100,255,60),(0,100,255,230))
            layer=Image.alpha_composite(layer.convert('RGBA'),overlay).convert('RGB');d=ImageDraw.Draw(layer)
        elif index==3 and arcs:
            for a,b,_ in arcs:
                d.line([point(a),point(b)],fill='#ff9c00',width=3)
        if route is not None:
            path(d,route,color)
        if index in (1,3) and not predicted:
            d.rectangle((0,side//2-20,side,side//2+20),fill='white')
            d.text((15,side//2-10),'No recorded solver path',font=font(20),fill='#b00020')
        if index==3 and arcs is None:
            d.rectangle((0,0,side,25),fill='white')
            d.text((8,4),'Graph not recorded (e.g. legacy matrix)',font=font(13),fill='#34495e')
        for name,rc,fill in [('S',source,'#1263d6'),('G',target,'#b65000')]:
            px,py=point(rc);radius=max(8,min(12,side/cols*.38))
            d.ellipse((px-radius,py-radius,px+radius,py+radius),fill=fill,outline='white',width=2)
            d.text((px-5,py-8),name,font=font(13),fill='white')
        canvas.paste(layer,(x,y+top))
        draw.text((x+8,y+top+side+12),caption,font=font(12),fill='#34495e')
    return canvas


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--outcomes',type=Path,required=True,help='outcomes.jsonl, before.jsonl, distilled.jsonl or after.jsonl')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['train','val','test'],default='val')
    p.add_argument('--mode',default='full',help='Recorded mode; use all to include every mode')
    p.add_argument('--limit',type=int,default=0,help='0 renders all matching records, including failures')
    p.add_argument('--panel-size',type=int,default=480)
    args=p.parse_args()
    if args.limit<0 or args.panel_size<320:
        p.error('limit must be nonnegative and panel-size >= 320')
    if args.output.exists() and any(args.output.iterdir()):
        p.error('output directory must be new or empty')
    config=json.loads((args.data/'dataset.json').read_text())
    land=config.get('navigation_protocol')==LAND_PROTOCOL
    folder=args.data/args.split
    records={r['sample_id']:r for r in (json.loads(s) for s in (folder/'tasks.jsonl').read_text().splitlines() if s.strip())}
    outcomes=[json.loads(s) for s in args.outcomes.read_text().splitlines() if s.strip()]
    outcomes=[r for r in outcomes if args.mode=='all' or r.get('mode')==args.mode]
    if args.limit:outcomes=outcomes[:args.limit]
    if not outcomes:p.error('no matching outcomes; check --mode')
    args.output.mkdir(parents=True,exist_ok=True)
    cards=[]; audit=[]
    for i,row in enumerate(outcomes):
        sid=row['sample_id']
        if sid not in records:raise ValueError(f'{sid}: missing from {args.split} manifest; check --data/--split')
        task=records[sid]; name=task['map_file']
        if Path(name).name!=name:raise ValueError('map_file must be a basename')
        with np.load(folder/name,allow_pickle=False) as data:
            rgb=data['image']; costs=data['true_costs']; ti=task['target_index']; si=task['source_index']
            source=data['sources_rc'][ti,si].tolist();target=data['targets_rc'][ti].tolist()
            gt=data['gt_path_rc'][ti,si,:int(data['gt_path_length'][ti,si])].tolist()
            optimum=float(data['optimal_cost'][ti,si])
            allowed=data['allowed_mask'] if land else None
            shortest_length=int(data['shortest_path_length'][ti,si]) if land else None
        if source!=task['source_rc'] or target!=task['target_rc']:raise ValueError(f'{sid}: manifest endpoints mismatch')
        if hashlib.sha256(rgb.tobytes()).hexdigest()!=task['map_sha256']:raise ValueError(f'{sid}: image checksum mismatch')
        if row.get('gt_path_rc') is not None and row['gt_path_rc']!=gt:raise ValueError(f'{sid}: log GT differs from dataset')
        validate_path(gt,source,target,costs.shape,allowed,no_corner_cut=land)
        predicted=row.get('path_rc')
        measured=None;violation=None
        if predicted is not None:
            # Invalid but actually selected paths must remain visible.
            validate_path(predicted,source,target,costs.shape)
            try:
                measured=evaluate_path(predicted,costs,source,target,optimum,allowed=allowed,
                                       no_corner_cut=land,shortest_length=shortest_length)
            except ValueError as exc:
                if row['status']!='constraint_failure':raise
                violation=str(exc)
            if violation is not None:
                measured=dict(true_cost=None,spl=0.,cost_efficiency=0.,constraint_violation=violation)
            if row.get('true_cost') is not None and not np.isclose(measured['true_cost'],row['true_cost']):
                raise ValueError(f'{sid}: recorded cost differs from dataset')
        arcs=row.get('compiled_graph',{}).get('arcs')
        if arcs is not None:
            for a,b,c in arcs:
                # Compiled arcs must be actual neighboring terrain cells.
                validate_path([a,b],a,b,costs.shape)
        title=f"{sid} | {row.get('mode','unknown')} | {row['status']}"
        note=f"S={source} G={target} | GT cost={optimum:.2f}"
        if violation:note+=' | INVALID: '+violation
        elif measured:note+=f" | Path cost={measured['true_cost']:.2f} | Regret={measured['relative_regret']:.3f}"
        image=render(dict(rgb=rgb,shape=costs.shape),source,target,predicted,gt,arcs,title,note,args.panel_size,
            region_model=row.get('compiled_graph') if row.get('model_representation')=='regions' else None,land=land)
        filename=f'{i:04d}_paths.png';image.save(args.output/filename)
        detail=dict(sample_id=sid,mode=row.get('mode'),status=row['status'],source_rc=source,target_rc=target,
                    image_file=filename,optimal_cost=optimum,measured=measured,error=row.get('error'),
                    path_rc=predicted,gt_path_rc=gt,compiled_graph=row.get('compiled_graph'),
                    model_answer=row.get('model_answer'),selected_indices=row.get('selected_indices'))
        (args.output/f'{i:04d}_details.json').write_text(json.dumps(detail,indent=2),encoding='utf-8')
        audit.append({k:v for k,v in detail.items() if k not in ('model_answer','compiled_graph','path_rc','gt_path_rc')})
        cards.append(f'<article><h2>{html.escape(title)}</h2><p>{html.escape(note)}</p><a href="{filename}"><img src="{filename}" loading="lazy"></a><p><a href="{i:04d}_details.json">Coordinates and model answer</a></p></article>')
        print(f'{i+1}/{len(outcomes)} {sid} -> {filename}')
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Recorded planning paths</title><style>body{font:16px sans-serif;background:#edf1f5;margin:24px}article{max-width:1100px;margin:24px auto}img{width:100%}h2{font-size:20px}</style><h1>Recorded Qwen + solver paths vs dataset GT</h1><p>Offline visualization only. No model inference, rerouting or route repair. Pink: solver path. Cyan: reference under current dataset rules. Orange: proposed graph or regions. Land protocol forbids lake/ocean; wall retains its finite cost. Failed predicted routes are shown, never repaired.</p>'+''.join(cards),encoding='utf-8')
    (args.output/'manifest.json').write_text(json.dumps(dict(outcomes=str(args.outcomes),dataset=config['dataset'],split=args.split,samples=audit),indent=2),encoding='utf-8')
    print(f'Open {args.output / "index.html"}')


if __name__=='__main__':main()
