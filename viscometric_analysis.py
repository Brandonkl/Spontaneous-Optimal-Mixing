import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as colors
import matplotlib.cm as cm
from skimage import measure

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
                
    if bc_label == 'posts':
        for x in range(Lx):
                for y in range(Ly): sim_points.add((x,y))
                    
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
                    bound[0, x, y, 0] = -round((x-pt[0])/((x-pt[0])**2 + (y-pt[1])**2)**(0.5), 4)
                    bound[0, x, y, 1] = -round((y-pt[1])/((x-pt[0])**2 + (y-pt[1])**2)**(0.5), 4) 
                    outer_bound.add((x,y))
            # in bound - non outer boundary points points with neighbors that are in outer boundary are inner boundary
            for x,y in in_r:
                if {(x+1,y),(x-1,y),(x,y+1),(x,y-1)} & outer_bound and (x,y) not in outer_bound:
                    bound[1, x, y, 0] = -round((x-pt[0])/((x-pt[0])**2 + (y-pt[1])**2)**(0.5), 4)
                    bound[1, x, y, 1] = -round((y-pt[1])/((x-pt[0])**2 + (y-pt[1])**2)**(0.5), 4)

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
threshold = 0.1


set_boundary(bound, sim_points, bc_label)
out_dirs = ["5_2fa","6_2fa","7_2fa", "8_2fa", "9_2fa", "10_2fa"]
for out_dir in out_dirs:
    braids = []
    T = int(1e10)
    vscale = 4
    qscale = 3
    t = 1
    mask = np.zeros((Lx,Ly))
    for x,y in sim_points: mask[x,y] = 1
    det_avg = np.zeros((Lx,Ly))
    omega_avg = np.zeros((Lx,Ly))
    det_std = [[[] for _ in range(Lx)] for _ in range(Ly)]
    omega_std = [[[] for _ in range(Lx)] for _ in range(Ly)]

    for rr,dd,ff in os.walk('./'):
        for dir in dd:
            if dir != out_dir: continue
            os.system(f"mkdir {out_dir}/u_mask")
            for root, dirs, files in os.walk(dir+'/u/'):
                for ut in sorted(files)[50:-1]:
                    f = open(dir+'/u/'+ut,'r').readlines()
                    f1 = open(dir+'/Q/Q'+ut[1:],'r').readlines()
                    if len(f) != Lx*Ly:
                        print("Error: u does not match provided system size")
                        break
                    ind = 0
                    u = np.empty((Lx,Ly,2))
                    det = np.empty((Lx,Ly))
                    omega = np.empty((Lx,Ly))
                    for x in range(Lx):
                        for y in range(Ly):
                            u[x,y] = f[ind].split()
                            ind += 1
                    ind = 0
                    Q = np.empty((Lx,Ly,2))
                    ss = np.empty((Lx,Ly))
                    for x in range(Lx):
                        for y in range(Lx):
                            Q[x,y] = f1[ind].split()
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
                    for x in range(Lx):
                        for y in range(Ly):
                            xup = (x + 1) % Lx
                            xdn = (x - 1)
                            yup = (y + 1) % Ly
                            ydn = (y - 1) 
                            det11 = u[xup,y,0] - u[xdn,y,0]
                            det12 = u[xup,y,1] - u[xdn,y,1]
                            det21 = u[x,yup,0] - u[x,ydn,0]
                            det22 = u[x,yup,1] - u[x,ydn,1]
                            det[x,y] = det11*det22 - det21*det12 
                            det_avg[x,y] += det[x,y]
                            omega[x,y] = det12 - det21
                            omega_avg[x,y] += omega[x,y]
                            det_std[x][y].append(det[x,y])
                            omega_std[x][y].append(omega[x,y])
                    det_std_dat = np.zeros((Lx,Ly))
                    omega_std_dat = np.zeros((Lx,Ly))
                    for x in range(Lx):
                        for y in range(Ly): 
                            det_std_dat[x,y] = np.std(det_std[x][y])
                            omega_std_dat[x,y] = np.std(omega_std[x][y])
                    qcontours = measure.find_contours(det,0.)
                    vcontours = measure.find_contours(omega,0.)
                    fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2,3)

                    im1 = ax1.imshow(omega, cmap='seismic', alpha=mask)
                    im2 = ax2.imshow(omega_avg*1000/t, cmap='seismic', alpha=mask)
                    im3 = ax3.imshow(omega_std_dat, cmap='Purples', alpha=mask)
                    im4 = ax4.imshow(det, cmap='PiYG', alpha=mask)
                    im5 = ax5.imshow(det_avg*1000/t, cmap='PiYG', alpha = mask)
                    im6 = ax6.imshow(det_std_dat, cmap='Purples', alpha=mask)
                    
                    for defect in braids[t // 1000]:
                        ax1.scatter(defect[1],defect[0],color = 'black')
                        ax4.scatter(defect[1],defect[0],color = 'black')
                    #for n, contour in enumerate(vcontours): ax1.plot(contour[:, 1], contour[:, 0], linewidth=2, color='black')
                    
                    for n, contour in enumerate(qcontours):
                        if len(contour) > 30: ax4.plot(contour[:, 1], contour[:, 0], linewidth=2, color='black')
                    
                    c2 = plt.colorbar(im2)
                    c3 = plt.colorbar(im3)
                    c5 = plt.colorbar(im5)
                    c6 = plt.colorbar(im6)

                    c2.mappable.set_clim(vmin=-vscale, vmax=vscale)
                    c3.mappable.set_clim(vmin=0, vmax=vscale)
                    c5.mappable.set_clim(vmin=-qscale, vmax=qscale)
                    c6.mappable.set_clim(vmin=0, vmax=qscale)

                    ax1.set_title(r"$\omega$")
                    ax2.set_title(r"$\langle \omega \rangle_t$")
                    ax3.set_title(r"$\sigma_{\omega}$")
                    ax4.set_title(r"$\mathcal{Q}$")
                    ax5.set_title(r"$\langle \mathcal{Q} \rangle_t$")
                    ax6.set_title(r"$\sigma_{\mathcal{Q}}$")

                    ax1.set_xticks([])
                    ax1.set_yticks([])
                    ax2.set_xticks([])
                    ax2.set_yticks([])
                    ax3.set_xticks([])
                    ax3.set_yticks([])
                    ax4.set_xticks([])
                    ax4.set_yticks([])
                    ax5.set_xticks([])
                    ax5.set_yticks([])
                    ax6.set_xticks([])
                    ax6.set_yticks([])
                    ax1.set_frame_on(False)
                    ax2.set_frame_on(False)
                    ax3.set_frame_on(False)
                    ax4.set_frame_on(False)
                    ax5.set_frame_on(False)
                    ax6.set_frame_on(False)

                    plt.savefig(f"{out_dir}/u_mask/t_"+'0'*(len(str(T-1)) - len(str(t)))+f"{t}.png")
                    plt.close()
                    t +=1000

    os.system(f"ffmpeg -framerate 15 -pattern_type glob -y -i '{out_dir}/u_mask/*.png'   -c:v libx264 -pix_fmt yuv420p {out_dir}/visco_n.mp4")
    os.system(f"rm -r {out_dir}/u_mask")
