import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as colors
import matplotlib.cm as cm
plt.rcParams["font.family"] = "Times New Roman"

def count(ss,c,locs, Lx,Ly):

    def cluster(ss,c,sign,locs, x,y):
        if c[x,y] == 0 and sign*ss[x,y] > 0 :
            c[x,y] = sign
            locs[-1].append((x,y))
            return cluster(ss,c,sign,locs,x+1,y) + cluster(ss,c,sign,locs,x-1,y) + cluster(ss,c,sign,locs,x,y-1) + cluster(ss,c,sign,locs,x,y+1) + cluster(ss,c,sign,locs,x+1,y-1) + cluster(ss,c,sign,locs,-1,y+1) + cluster(ss,c,sign,locs,x-1,y-1) + cluster(ss,c,sign,locs,x+1,y+1)
        return 0

    charges = []
    for x in range(Lx):
        for y in range(Ly):
            if ss[x,y] == 0: continue
            elif c[x,y] == 0:
                charges.append(-int(ss[x,y]/abs(ss[x,y])))
                locs.append([])
                cluster(ss,c,ss[x,y],locs,x,y)
    return charges

def avgLoc(s):
    (xavg, yavg) = (0,0)
    for p in s: (xavg, yavg) = np.add((xavg,yavg),p)
    return (xavg/len(s), yavg/len(s))

def set_boundary(bound, sim_points, bc_label):
    
    outer_bound = set()
    b = set()
    
    if bc_label == 'circular':
        radius = Lx//2 - 1
        
        # marks in bound points
        for x in range(Lx):
            for y in range(Ly):
                if (x-radius)**2 + (y-radius)**2 <= radius**2: sim_points.add((x, y))
        # in bound points that border out of bound points are outer boundary
        for x,y in sim_points:
            if {(x+1,y),(x-1,y),(x,y+1),(x,y-1)} - sim_points: 
                bound[1, x, y, 0] = round((x-radius)/((x-radius)**2 + (y-radius)**2)**(0.5), 4)
                bound[1, x, y, 1] = round((y-radius)/((x-radius)**2 + (y-radius)**2)**(0.5), 4) 
                outer_bound.add((x,y))
        # in bound - non outer boundary points points with neighbors that are in outer boundary are inner boundary
        for x,y in sim_points:
            if {(x+1,y),(x-1,y),(x,y+1),(x,y-1)} & outer_bound and (x,y) not in outer_bound:
                bound[0, x, y, 0] = round((x-radius)/((x-radius)**2 + (y-radius)**2)**(0.5), 4)
                bound[0, x, y, 1] = round((y-radius)/((x-radius)**2 + (y-radius)**2)**(0.5), 4)
                b.add((x,y))
                
    if bc_label == 'posts':
        for x in range(Lx):
                for y in range(Ly): sim_points.add((x,y))
        ibx = []
        iby = []
        ibnx = []
        ibny = []
        obx = []
        oby = []
        obnx = []
        obny = []
                    
        post_list = [(Lx-50, Ly-50, 15), (Lx-50, 50, 15), (50, Ly-50, 15), (50, 50, 15)] # center_x, center_y, r
        for pt in post_list:
            in_r = set()
            # marks in bound points
            for x in range(Lx):
                for y in range(Ly):
                    if (x-pt[0])**2 + (y-pt[1])**2 <= pt[2]**2: 
                        in_r.add((x, y))
                        sim_points.remove((x,y))
            # in bound points that border out of bound points are outer boundary
            for x,y in in_r:
                if {(x+1,y),(x-1,y),(x,y+1),(x,y-1)} - in_r: 
                    ibx.append(x)
                    iby.append(y)
                    bound[0, x, y, 0] = -round((x-pt[0])/((x-pt[0])**2 + (y-pt[1])**2)**(0.5), 4)
                    bound[0, x, y, 1] = -round((y-pt[1])/((x-pt[0])**2 + (y-pt[1])**2)**(0.5), 4) 
                    ibnx.append(bound[0, x, y, 0])
                    ibny.append(bound[0, x, y, 1])
                    outer_bound.add((x,y))
            # in bound - non outer boundary points points with neighbors that are in outer boundary are inner boundary
            for x,y in in_r:
                if {(x+1,y),(x-1,y),(x,y+1),(x,y-1)} & outer_bound and (x,y) not in outer_bound:
                    obx.append(x)
                    oby.append(y)
                    bound[1, x, y, 0] = -round((x-pt[0])/((x-pt[0])**2 + (y-pt[1])**2)**(0.5), 4)
                    bound[1, x, y, 1] = -round((y-pt[1])/((x-p[0])**2 + (y-pt[1])**2)**(0.5), 4)
                    obnx.append(bound[1, x, y, 0])
                    obny.append(bound[1, x, y, 1])
                    
                    
        import matplotlib.pyplot as plt
        plt.quiver(ibx, iby, ibnx, ibny, color = 'red')
        plt.quiver(obx,oby,obnx,obny,color='blue')

    if bc_label == 'tanh':
        k = 1.1 # between 1 an and 2.pertrusion sharpness (cusp at 1 ~ circular at 2)
        b = 1  # between 1 and Lx. petrusion dispersion
        n = 1   # integer. number of cusps.     
        radius = Lx//2 - 1
        # marks in bound points
        for x in range(Lx):
            for y in range(Ly):
                if ((x-radius)**2 + (y-radius)**2)**0.5 <= radius * np.tanh(b*(k - np.cos(n*np.arctan2(y-radius,x-radius)))): sim_points.add((x, y))
        # in bound points that border out of bound points are outer boundary
        for x,y in sim_points:
            if {(x+1,y),(x-1,y),(x,y+1),(x,y-1)} - sim_points: 
                outer_bound.add((x,y))
                t = np.arctan2(y-radius,x-radius)
                s = np.sin(t)
                c = np.cos(t)
                r = np.tanh(b*(k - np.cos(n*t)))
                dr = (1 - (np.tanh(b*(k - np.cos(n*t))))**2)*np.sin(n*t)*n*b
                norm = (r**2 + dr**2)**(0.5)
                bound[1, x, y, 0] = round((c*r + s*dr)/norm, 4)
                bound[1, x, y, 1] = round((s*r - c*dr)/norm, 4)
        # in bound - non outer boundary points points with neighbors that are in outer boundary are inner boundary
        for x,y in sim_points:
            if {(x+1,y),(x-1,y),(x,y+1),(x,y-1)} & outer_bound and (x,y) not in outer_bound:
                t = np.arctan2(y-radius,x-radius)
                s = np.sin(t)
                c = np.cos(t)
                r = np.tanh(b*(k - np.cos(n*t)))
                dr = (1 - (np.tanh(b*(k - np.cos(n*t))))**2)*np.sin(n*t)*n*b
                norm = (r**2 + dr**2)**(0.5)
                bound[0, x, y, 0] = round((c*r + s*dr)/norm, 4)
                bound[0, x, y, 1] = round((s*r - c*dr)/norm, 4)
    
    if bc_label == 'cardioid':
        
        for xs in range(Lx):
            for ys in range(Ly):
                x = xs - Lx//4
                y = ys - Ly//2
                if (x**2 + y**2)**0.5 <= (Lx//3)*(1 + x/(x**2 + y**2 + 1e-10)**0.5): sim_points.add((xs, ys))
        # in bound points that border out of bound points are outer boundary
        for x,y in sim_points:
            if {(x+1,y),(x-1,y),(x,y+1),(x,y-1)} - sim_points: 
                outer_bound.add((x,y))
                if x == Lx//4 and y == Ly//2: bound[1, x, y, 0] = -1
                else:
                    a = Lx//3
                    c = (x - Lx//4)/((x - Lx//4)**2 + (y-Ly//2)**2)**(0.5)
                    s = (y - Ly//2)/((x - Lx//4)**2 + (y-Ly//2)**2)**(0.5)
                    n = ((x - Lx//4)**2 + (y-Ly//2)**2 + (a*s)**2)**(0.5)
                    bound[1, x, y, 0] = round((x - Lx//4 - a*s**2)/n, 4)
                    bound[1, x, y, 1] = round((y - Ly//2 + a*s*c)/n, 4)
        # in bound - non outer boundary points points with neighbors that are in outer boundary are inner boundary
        for x,y in sim_points:
            if {(x+1,y),(x-1,y),(x,y+1),(x,y-1)} & outer_bound and (x,y) not in outer_bound:
                if x == Lx//4 + 1 and y == Ly//2: bound[0, x, y, 0] = -1
                else:
                    a = Lx//3
                    c = (x - Lx//4)/((x - Lx//4)**2 + (y-Ly//2)**2)**(0.5)
                    s = (y - Ly//2)/((x - Lx//4)**2 + (y-Ly//2)**2)**(0.5)
                    n = ((x - Lx//4)**2 + (y-Ly//2)**2 + (a*s)**2)**(0.5)
                    bound[0, x, y, 0] = round((x - Lx//4 - a*s**2)/n, 4)
                    bound[0, x, y, 1] = round((y - Ly//2 + a*s*c)/n, 4)
                    
    if bc_label == 'no_slip_channel':
        bound[0,:,1, 0] = 0
        bound[0,:,1, 1] = -1 # inner boundary bottom
        bound[0,:,-2, 0] = 0
        bound[0,:,-2, 1] = 1 # inner boundary top
        bound[1,:,0, 0] = 0
        bound[1,:,0, 1] = -1 # outer boundary bottom
        bound[1,:,-1, 0] = 0
        bound[1,:,-1, 1] = 1 # outer boundary top
        for x in range(Lx):
            for y in range(Ly): sim_points.add((x,y))
                
    if bc_label.endswith('.txt'): 
        for x in range(Lx):
            for y in range(Ly): sim_points.add((x,y))

bc_label = 'circular'
sim_points = set()
Lx = Ly = 100
bound = np.zeros((2, Lx, Ly, 2))
x_bound = []
y_bound = []
if bc_label == 'circular':
    rad = Lx//2 - 1
    for theta in np.arange(0,2*np.pi, 0.01):
        x_bound.append(rad*np.cos(theta) + rad)
        y_bound.append(rad*np.sin(theta) + rad)
set_boundary(bound, sim_points, bc_label)
X, Y = np.meshgrid(np.arange(Lx),np.arange(Ly))
nres = 1
threshold = 0.1
plot = False
braids = []
out_dir = "5_2fa"

for rr,dd,ff in os.walk('./'):
    for dir in dd:
        if dir != out_dir: continue
        if plot: os.system(f"mkdir {out_dir}/Q_mask")
        for root, dirs, files in os.walk(dir+'/Q/'):
            for Qt in sorted(files)[40:140]:
                f = open(dir+'/Q/'+Qt,'r').readlines()
                if len(f) != Lx*Ly:
                    print("Error: Q does not match provided system size")
                    break
                ind = 0
                Q = np.empty((Lx,Ly,2))
                ss = np.empty((Lx,Ly))
                for x in range(Lx):
                    for y in range(Lx):
                        Q[x,y] = f[ind].split()
                        ind += 1
                for x in range(Lx):
                    for y in range(Lx):
                        xup = (x + 1) % Lx
                        xdn = (x - 1)
                        yup = (y + 1) % Ly
                        ydn = (y - 1) 
                        twice_dxQxx = Q[xup,y,0] - Q[xdn,y,0]
                        twice_dxQxy = Q[xup,y,1] - Q[xdn,y,1]
                        twice_dyQxx = Q[x,yup,0] - Q[x,ydn,0]
                        twice_dyQxy = Q[x,yup,1] - Q[x,ydn,1]
                        ss[x,y] = twice_dxQxy*twice_dyQxx - twice_dxQxx*twice_dyQxy 
                        if bound[0, x, y, 0] or bound[0, x, y, 1] or bound[1, x, y, 0] or bound[1, x, y, 1]: ss[x,y] = 0
                        if (x,y) not in sim_points: ss[x,y] = 0
                a = Q[:,:,0] # Qxx values
                b = -Q[:,:,1] # Qxy values
                denom = np.sqrt(b*b + (a+np.sqrt(a*a+b*b))**2) # normalization for n
                bad_ptsX, bad_ptsY = np.where(denom==0) # trick to avoid divisions by zero
                denom[bad_ptsX, bad_ptsY] = 1    
                nx = (a+np.sqrt(a*a+b*b))/denom # n_x values
                ny = b/denom # n_y values        
                nx[bad_ptsX, bad_ptsY] = 0
                ny[bad_ptsX, bad_ptsY] = 0
                defects = np.ma.masked_where(np.abs(ss) > threshold, ss).mask * ss
                c = np.zeros_like(defects)
                locs = []
                num = count(defects,c,locs,Lx,Ly)
                braids.append([avgLoc(l) for l in locs])
                mask = np.full((Lx,Ly), c.min())
                for x,y in sim_points: mask[x,y] = c[x,y]
                    
                if plot:
                    fig = plt.figure(figsize=(20, 20))
                    plt.quiver(X.T[::nres,::nres], Y.T[::nres,::nres], nx[::nres,::nres],ny[::nres,::nres])
                    plt.imshow(mask.T,vmin = c.min(), vmax = c.max(), interpolation='none')
                    plt.savefig(f"{out_dir}/Q_mask/{Qt[:-4]}_{sum(num)/2}.png",dpi=200)
                    plt.close(fig)

cutoff = 40
braids = braids[cutoff:]
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 2, projection='3d')
X = []
Y = []
Z = []
T = len(braids)
dim = len(braids[0])
colors = ['#1f78b4', '#33a02c', '#6a3d9a', '#e31a1c', '#4ea395', '#ff7f00', '#a65628','black', 'red', 'purple']

for b in range(dim):
    cx = braids[0][b][0]
    cy = braids[0][b][1]
    ct = cutoff*1000
    x = []
    y = []
    z = []
    for ts in braids:
        d = [(cx - p[0])**2 + (cy - p[1])**2 for p in ts]
        l = ts[d.index(min(d))]
        x.append(l[0])
        y.append(l[1])
        z.append(ct)
        cx = x[-1]
        cy = y[-1]
        ct += 1000
    ax1.plot(x,y,z, color = colors[b], linewidth=4)
    X.append(y)
    Y.append(x)
    Z.append(z)
ax1.set_box_aspect((1,1,14))
ax1.view_init(elev=10, azim=-15, roll=0)
# Make panes transparent
ax1.xaxis.pane.fill = False # Left pane
ax1.yaxis.pane.fill = False # Right pane

# Remove grid lines
ax1.grid(False)

# Remove tick labels
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_zticklabels([])

# Transparent spines
ax1.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax1.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax1.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

# Transparent panes
ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# No ticks
ax1.set_xticks([]) 
ax1.set_yticks([]) 
ax1.set_zticks([])

ax0 = fig.add_subplot(1, 2, 1)
ax0.set_aspect(1)
ax0.plot(x_bound, y_bound, color = 'black')
ax0.set_xlim(-1,Lx+1)
ax0.set_ylim(-1,Ly+1)
ax0.axis('off')
for i in range(dim): ax0.plot(X[i],Y[i], color = colors[i])

pltenv = False
if pltenv:
    q = 3/2
    env_x = []
    env_y = []
    for theta in np.arange(0,2*np.pi+0.01,.05):
        env_th = np.arctan2((2*q-1)*np.sin(theta)+np.sin((2*q-1)*theta),(2*q-1)*np.cos(theta)+np.cos((2*q-1)*theta))
        env_r = (Lx//2 - 1)*(1 + (2*q-1)/(2*q**2)*(np.cos(2*theta*(q-1))-1))**0.5
        env_x.append(env_r*np.cos(env_th) + Lx//2)
        env_y.append(env_r*np.sin(env_th) + Lx//2)
    ax0.plot(env_y,env_x,color = 'black', linestyle='--', linewidth=2)
plt.savefig(f"{out_dir}/trajectories.png", dpi=1600)
if pltenv: quit()
bd = []
td = []
yd = []
ccw = []
cc = []
braidword = open(f"{out_dir}/out.txt", "w")
os.system(f"mkdir {out_dir}/worldlines")
for t in range(T):
    u = [X[i][t] for i in range(dim)]
    uy = [Y[i][t] for i in range(dim)]
    if len(u) != len(set(u)) or len(uy) != len(set(uy)): continue # fixes numerical bug where defects have exact same x or y coordinate
    s = [u.index(b) for b in sorted(u)]
    uy = [uy[k] for k in s]
    if len(bd) == 0:
        bd.append(s)
        td.append(t*1000)
        yd.append(uy)
    if s != bd[-1]: # detects swaps
        so = bd[-1]
        bd.append(s)
        td.append(t*1000)
        yd.append(uy)
        swap = 0
        while swap < len(s):
            if so[swap] != s[swap]:
                if uy[swap] > uy[swap+1]: 
                    cc.append((u[s[swap]],td[-1]))
                    braidword.write(f"{swap},{swap+1},{td[-1]}: sigma_{swap+1}\n")
                else: 
                    ccw.append((u[s[swap]],td[-1]))
                    braidword.write(f"{swap},{swap+1},{td[-1]}: sigma_{swap+1}^{-1}\n")
                swap += 1
            swap += 1
braidword.close()

trace = True
for t in range(T):
    fig, axs = plt.subplots(1,2)
    for i in range(dim): 
        if trace: axs[0].plot(X[i][:t],Y[i][:t],color = colors[i])
        axs[0].plot([X[i][t]],[Y[i][t]], marker = 'o', markersize=5,color = colors[i])
        axs[0].plot(x_bound, y_bound, color = 'black')
        axs[1].plot(X[i][:t], Z[i][:t], color = colors[i])
    for p in cc: 
        if t*1000 < p[1]: break
        else: 
            axs[1].plot([p[0]],[p[1]+1000*cutoff],marker='o',markersize=5,color = 'black')
    for p in ccw: 
        if t*1000 < p[1]: break
        else: axs[1].plot([p[0]],[p[1]+1000*cutoff],marker='X',markersize=5,color = 'red')  
    axs[1].set_ylabel("Time")
    axs[1].set_xlabel("Defect Projection Along X")
    axs[0].set_xlim(-1,Lx+1)
    axs[0].set_ylim(-1,Ly+1)
    axs[1].set_xlim(-1,Lx+1)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].axis('off')
    axs[1].set_xticks([])
    axs[1].set_yticks([100000*i for i in range(T//100000 + 1)])
    axs[0].set_aspect(1)
    axs[1].set_aspect(0.0003)
    plt.savefig(f"{out_dir}/worldlines/t_"+'0'*(len(str(T-1)) - len(str(t)))+f"{t}")
    plt.close()
os.system(f"ffmpeg -framerate 15 -pattern_type glob -y -i '{out_dir}/worldlines/*.png'   -c:v libx264 -pix_fmt yuv420p {out_dir}/braid.mp4")
os.system(f"scp {out_dir}/worldlines/t_{T-1}.png {out_dir}")
os.system(f"rm -r {out_dir}/worldlines")
norms = []
for i in range(dim):
    nb = []
    for t in range(len(td)): 
        nb.append(bd[t].index(i))
    norms.append(nb)
    plt.figure()
    plt.plot(nb,td, color = colors[i])
    plt.gca().set_aspect(0.0003/40)
    plt.savefig(f"{out_dir}/{i}-place.png")
plt.figure()
for n in norms: plt.plot(n, range(len(n)), color = colors[norms.index(n)])
plt.savefig(f"{out_dir}/braid-norm.png")
