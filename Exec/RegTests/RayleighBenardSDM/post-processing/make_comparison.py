from pathlib import Path
import argparse
import csv
import math
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(
    description='Plot the statistical-efficiency and Hall superdroplet comparison.'
)
parser.add_argument('--statistical-efficiency-run', type=Path, required=True,
                    help='Run directory containing run.log and binned DSD files.')
parser.add_argument('--hall-run', type=Path, required=True,
                    help='Hall run directory containing run.log and binned DSD files.')
parser.add_argument('--output-dir', type=Path, default=SCRIPT_DIR,
                    help='Directory for the PNG and derived CSV files (default: this folder).')
args = parser.parse_args()
OUT = args.output_dir.expanduser().resolve()
OUT.mkdir(parents=True, exist_ok=True)
CASES = {
    'statistical_efficiency': args.statistical_efficiency_run.expanduser().resolve(),
    'Hall': args.hall_run.expanduser().resolve(),
}
TIMES = [500, 1000, 1500]
CELL_VOLUME = (0.2 / 64) * (0.2 / 64) * (0.1 / 32)
TAIL_EDGE_UM = 24.71
COLORS = {'statistical_efficiency': (205, 78, 64), 'Hall': (39, 112, 174)}

spectra = {}
summary = {}
collisions = {}
deposited = {}
active_sd = {}
for kernel, run in CASES.items():
    log = (run / 'run.log').read_text(errors='replace')
    coll_by_step = {}
    dep_by_step = {}
    active_by_step = {}
    step = None
    for line in log.splitlines():
        mi = re.search(r'iteration=(\d+).*evolving (\d+) super-droplets representing (\d+) particles', line)
        if mi:
            step = int(mi.group(1))
            active_by_step[step] = (int(mi.group(2)), int(mi.group(3)))
        mc = re.search(r'number of collisions\s*=\s*(\d+)', line, re.I)
        if mc and step is not None:
            coll_by_step[step] = int(mc.group(1))
        md = re.search(r'deactivated (\d+) super-droplets at boundaries', line, re.I)
        if md and step is not None:
            dep_by_step[step] = dep_by_step.get(step, 0) + int(md.group(1))
    for it in TIMES:
        collisions[(kernel, it)] = sum(v for k, v in coll_by_step.items() if k <= it)
        deposited[(kernel, it)] = sum(v for k, v in dep_by_step.items() if k <= it)
        active_sd[(kernel, it)] = active_by_step.get(it, (0, 0))[0]
        f = run / f'super_droplets_moisture_binned_dsd_{it:05d}.txt'
        data = np.loadtxt(f, max_rows=120)
        radius, g_mass, g_num = data.T
        dln = math.log(radius[1] / radius[0])
        # ERF's global histogram is a sum of per-cell concentrations. Restore
        # total counts by multiplying by the uniform cell volume.
        dndlnr = g_num * CELL_VOLUME
        bin_count = dndlnr * dln
        mass_bin_kg = g_mass * CELL_VOLUME * dln
        total = float(bin_count.sum())
        pdf = bin_count / total
        mean_radius = float((radius * bin_count).sum() / total)
        tail = float(bin_count[radius * 1e6 >= TAIL_EDGE_UM - 0.02].sum())
        spectra[(kernel, it)] = (radius * 1e6, dndlnr, bin_count)
        summary[(kernel, it)] = {
            'time_s': it / 1000,
            'kernel': kernel,
            'droplet_number': total,
            'mean_radius_um': mean_radius * 1e6,
            'water_mass_kg': float(mass_bin_kg.sum()),
            'number_radius_ge_24_71um': tail,
            'collision_events_cumulative': collisions[(kernel, it)],
            'plate_deactivated_superdroplets_cumulative': deposited[(kernel, it)],
            'active_superdroplets': active_sd[(kernel, it)],
        }

for it in TIMES:
    p = summary[('statistical_efficiency', it)]
    h = summary[('Hall', it)]
    ppdf = spectra[('statistical_efficiency', it)][2] / p['droplet_number']
    hpdf = spectra[('Hall', it)][2] / h['droplet_number']
    tv = float(0.5 * np.abs(ppdf - hpdf).sum())
    p['total_variation_vs_Hall'] = tv
    h['total_variation_vs_statistical_efficiency'] = tv

with (OUT / 'collision_efficiency_rerun_summary.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in summary.values() for k in row)))
    writer.writeheader()
    writer.writerows(summary.values())

with (OUT / 'compare_dsd_number_statistical_efficiency_vs_hall_rerun.csv').open('w', newline='') as f:
    fields = ['radius_um']
    for it in TIMES:
        for kernel in CASES:
            suffix = f'{kernel.lower()}_{it/1000:.1f}s'
            fields += [f'dN_dlnR_{suffix}', f'bin_count_{suffix}']
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for j, r_um in enumerate(spectra[('statistical_efficiency', TIMES[0])][0]):
        row = {'radius_um': r_um}
        for it in TIMES:
            for kernel in CASES:
                r, dndlnr, count = spectra[(kernel, it)]
                suffix = f'{kernel.lower()}_{it/1000:.1f}s'
                row[f'dN_dlnR_{suffix}'] = dndlnr[j]
                row[f'bin_count_{suffix}'] = count[j]
        writer.writerow(row)

# A clean raster figure, drawn at publication-friendly resolution with PIL.
W, H = 1900, 1370
im = Image.new('RGB', (W, H), 'white')
draw = ImageDraw.Draw(im)
font_paths = ['/System/Library/Fonts/Supplemental/Arial.ttf', '/System/Library/Fonts/Helvetica.ttc']
font_path = next((p for p in font_paths if Path(p).exists()), None)
def font(size):
    return ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
F_TITLE, F_SUB, F_PANEL, F_AXIS, F_SMALL = font(34), font(19), font(21), font(17), font(15)
BLACK = (32, 40, 48)
GRAY = (110, 120, 130)
GRID = (226, 231, 236)

# Title and setup line
draw.text((80, 38), 'Statistical-efficiency kernel vs Hall kernel', fill=BLACK, font=F_TITLE)
draw.text((82, 88), 'Rayleigh-Benard plates with deposition  |  64x64x32  |  DeltaT = 1 K  |  epsilon = 1e-2 m^2/s^3  |  Re_lambda = 2500', fill=GRAY, font=F_SUB)
draw.text((82, 117), 'Periodic x/y, no-slip fixed-temperature plates in z  |  same fixed seed and 100 droplets/cm^3 initial', fill=GRAY, font=F_SUB)

panels = [(80, 175, 820, 455), (1000, 175, 820, 455), (80, 690, 820, 455), (1000, 690, 820, 455)]
def panel_frame(rect, title, subtitle=''):
    x, y, w, h = rect
    draw.rounded_rectangle((x, y, x+w, y+h), radius=14, outline=(202, 210, 218), width=2, fill=(252, 253, 254))
    draw.text((x+24, y+17), title, fill=BLACK, font=F_PANEL)
    if subtitle:
        draw.text((x+24, y+45), subtitle, fill=GRAY, font=F_SMALL)
    return (x+86, y+84, w-115, h-150)

def legend(x, y, entries):
    for color, label in entries:
        draw.line((x, y+9, x+34, y+9), fill=color, width=4)
        draw.ellipse((x+13, y+4, x+23, y+14), fill=color)
        draw.text((x+43, y), label, fill=BLACK, font=F_SMALL)
        x += 280

def draw_dsd(rect, it, title):
    plot = panel_frame(rect, title, f'Exact binned number DSD at t = {it/1000:.1f} s; log y-scale')
    x0, y0, pw, ph = plot
    xmin, xmax, ymin, ymax = 14.0, 30.0, 1.0, 3.0e6
    def xp(v): return x0 + (v-xmin)/(xmax-xmin)*pw
    def yp(v): return y0 + ph - (math.log10(max(ymin, v))-math.log10(ymin))/(math.log10(ymax)-math.log10(ymin))*ph
    for val in [1,10,100,1000,10000,100000,1000000]:
        yy = yp(val)
        draw.line((x0, yy, x0+pw, yy), fill=GRID, width=1)
        draw.text((x0-66, yy-9), f'{val:g}', fill=GRAY, font=F_SMALL)
    for val in [14,16,18,20,22,24,26,28,30]:
        xx = xp(val)
        draw.line((xx, y0, xx, y0+ph), fill=(241,244,247), width=1)
        draw.text((xx-12, y0+ph+13), str(val), fill=GRAY, font=F_SMALL)
    draw.line((x0, y0, x0, y0+ph), fill=BLACK, width=2)
    draw.line((x0, y0+ph, x0+pw, y0+ph), fill=BLACK, width=2)
    draw.text((x0+pw/2-50, y0+ph+39), 'Radius (um)', fill=BLACK, font=F_AXIS)
    draw.text((x0-70, y0-24), 'dN/dlnR', fill=BLACK, font=F_AXIS)
    legend(rect[0]+rect[2]-390, rect[1]+18, [(COLORS['statistical_efficiency'], 'Statistical-efficiency kernel'), (COLORS['Hall'], 'Hall kernel')])
    for kernel in CASES:
        r, vals, _ = spectra[(kernel, it)]
        pts = [(xp(float(rv)), yp(float(v))) for rv, v in zip(r, vals) if xmin <= rv <= xmax and v > 0]
        if len(pts) > 1:
            draw.line(pts, fill=COLORS[kernel], width=4, joint='curve')
        for xx, yy in pts:
            draw.ellipse((xx-4, yy-4, xx+4, yy+4), fill=COLORS[kernel])

draw_dsd(panels[0], 500, 'A. Bulk spectrum at 0.5 s')
draw_dsd(panels[1], 1500, 'B. Bulk spectrum at 1.5 s')

# Cumulative collision events
plot = panel_frame(panels[2], 'C. Cumulative collision events', 'The bulk DSD hides a clear kernel-rate difference')
x0,y0,pw,ph = plot
xs = [0.0] + [it/1000 for it in TIMES]
for val in [0,50,100,150,200]:
    yy = y0+ph-val/200*ph
    draw.line((x0,yy,x0+pw,yy), fill=GRID, width=1)
    draw.text((x0-47, yy-9), str(val), fill=GRAY, font=F_SMALL)
for val in [0,.5,1.0,1.5]:
    xx=x0+val/1.5*pw
    draw.line((xx,y0,xx,y0+ph), fill=(241,244,247), width=1)
    draw.text((xx-12,y0+ph+13),f'{val:g}',fill=GRAY,font=F_SMALL)
draw.line((x0,y0,x0,y0+ph),fill=BLACK,width=2); draw.line((x0,y0+ph,x0+pw,y0+ph),fill=BLACK,width=2)
draw.text((x0+pw/2-34,y0+ph+39),'Time (s)',fill=BLACK,font=F_AXIS)
draw.text((x0-55,y0-24),'Events',fill=BLACK,font=F_AXIS)
legend(panels[2][0]+panels[2][2]-390,panels[2][1]+18,[(COLORS['statistical_efficiency'],'Statistical-efficiency kernel'),(COLORS['Hall'],'Hall kernel')])
for kernel in CASES:
    ys=[0]+[collisions[(kernel,it)] for it in TIMES]
    pts=[(x0+t/1.5*pw,y0+ph-v/200*ph) for t,v in zip(xs,ys)]
    draw.line(pts,fill=COLORS[kernel],width=4,joint='curve')
    for xx,yy in pts: draw.ellipse((xx-5,yy-5,xx+5,yy+5),fill=COLORS[kernel])
final_p=collisions[('statistical_efficiency',1500)]; final_h=collisions[('Hall',1500)]
draw.text((x0+22,y0+18),f'1.5 s total: {final_p} statistical-efficiency vs {final_h} Hall ({final_h/final_p:.2f}×)',fill=BLACK,font=F_AXIS)

# Upper-tail counts, where the difference is visible
plot = panel_frame(panels[3], 'D. Upper-tail droplets retained', f'Number in bins beginning at radius >= {TAIL_EDGE_UM:.2f} um; deposition depletes this tail')
x0,y0,pw,ph=plot
vals_max=150
for val in [1,10,100]:
    yy=y0+ph-(math.log10(val)-0)/(math.log10(vals_max)-0)*ph
    draw.line((x0,yy,x0+pw,yy),fill=GRID,width=1)
    draw.text((x0-42,yy-9),str(val),fill=GRAY,font=F_SMALL)
draw.line((x0,y0,x0,y0+ph),fill=BLACK,width=2); draw.line((x0,y0+ph,x0+pw,y0+ph),fill=BLACK,width=2)
draw.text((x0+pw/2-52,y0+ph+39),'Time (s)',fill=BLACK,font=F_AXIS)
draw.text((x0-75,y0-24),'Droplet count',fill=BLACK,font=F_AXIS)
centers=[x0+pw*(i+0.5)/3 for i in range(3)]
for i,it in enumerate(TIMES):
    for j,kernel in enumerate(CASES):
        count=summary[(kernel,it)]['number_radius_ge_24_71um']
        value=max(1.0,count)
        xx=centers[i]+(-15 if j==0 else 15)
        yy=y0+ph-math.log10(value)/math.log10(vals_max)*ph
        draw.line((xx,yy,xx,y0+ph),fill=COLORS[kernel],width=17)
        draw.text((xx-20,yy-25),f'{count:.0f}',fill=COLORS[kernel],font=F_SMALL)
    draw.text((centers[i]-24,y0+ph+13),f'{it/1000:.1f}',fill=GRAY,font=F_SMALL)
legend(panels[3][0]+panels[3][2]-390,panels[3][1]+18,[(COLORS['statistical_efficiency'],'Statistical-efficiency kernel'),(COLORS['Hall'],'Hall kernel')])

footer = 'By 1.5 s, plate deposition deactivated 180,214 / 180,236 super-droplets (statistical-efficiency / Hall). Global number-spectrum TV distance at 1.5 s: 0.023%.'
draw.text((82, 1190), footer, fill=GRAY, font=F_SUB)
draw.text((82, 1222), 'Conclusion: collision rates differ, but bulk spectra remain nearly identical because only 66 vs 167 collision events occur before most particles deposit.', fill=BLACK, font=F_SUB)
draw.text((82, 1253), 'Binned spectra use 120 logarithmic bins from 5 um to 2 mm. Initial droplets: 16-24 um log-normal, 1e8 m^-3; phase change and LES disabled.', fill=GRAY, font=F_SMALL)

im.save(OUT / 'compare_dsd_number_statistical_efficiency_vs_hall_rerun.png', dpi=(180,180))
print('wrote', OUT / 'compare_dsd_number_statistical_efficiency_vs_hall_rerun.png')
for it in TIMES:
    p=summary[('statistical_efficiency',it)]; h=summary[('Hall',it)]
    print(f"t={it/1000:.1f}s: N={p['droplet_number']:.0f}/{h['droplet_number']:.0f}, collisions={p['collision_events_cumulative']}/{h['collision_events_cumulative']}, plate_deactivated={p['plate_deactivated_superdroplets_cumulative']}/{h['plate_deactivated_superdroplets_cumulative']}, tail={p['number_radius_ge_24_71um']:.0f}/{h['number_radius_ge_24_71um']:.0f}, TV={p['total_variation_vs_Hall']*100:.5f}%")
