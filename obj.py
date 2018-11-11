import numpy as np
import os
from time import time

def PrintTime(t1,tag):
    t2=time()
    dt=(t2-t1)*1000
    print("%s: %dms\n"%(tag,dt))

def normlize_vec(fn):
    l=np.sqrt(np.sum(fn*fn,axis=1))
    l=np.expand_dims(l, 1)
    print('l:', l.shape)

    invliadId = l[:,0] < 0.00001
    l[invliadId,:]=1
    fn=fn/l

    return fn

def cal_fn(v,f):
    v1=v[f[:,0],:]
    v2 = v[f[:, 1], :]
    v3 = v[f[:, 2], :]
    fn=np.cross(v2-v1,v3-v1)
    fn=normlize_vec(fn)
    print('fn:',fn.shape)

    return fn

def cal_vn(v,f):
    t1=time()
    fn=cal_fn(v,f)
    vn=np.zeros((v.shape[0],3),dtype=np.float)
    for i in range(f.shape[0]):
        tn = fn[i, :]
        i1 = f[i, 0]
        i2 = f[i, 1]
        i3 = f[i, 2]
        vn[i1, :] += tn
        vn[i2, :] += tn
        vn[i3, :] += tn
    vn=normlize_vec(vn)
    PrintTime(t1, "cal_vn")
    print('vn:',vn.shape)

    return vn

def load_obj(name):
    t1=time()
    with open(name,'r') as fid:
        lines=fid.readlines()
        vs=[]
        fs=[]
        for line in lines:
            tag=line.split(' ')[0]
            if tag not in ['v','f']:
                continue

            x = line.split(' ')[1]
            y = line.split(' ')[2]
            z = line.split(' ')[3]

            if tag=='v' :
                vs.append(float(x))
                vs.append(float(y))
                vs.append(float(z))
            elif tag=='f':
                fs.append(int(x))
                fs.append(int(y))
                fs.append(int(z))
        print("#v: %d ,#f: %d " % (len(vs)/3,len(fs)/3))

        v=np.asarray(vs,dtype=np.float).reshape(-1,3)
        f=np.asarray(fs,dtype=np.int).reshape(-1,3) -1
        PrintTime(t1,"load_obj")

        return v,f

def save_obj(v,vn,f,name):
    with open(name,'w') as fid:
        for i in range(v.shape[0]):
            fid.write('v %f %f %f\n' %(v[i,0],v[i,1],v[i,2]))
        for i in range(vn.shape[0]):
            fid.write('vn %f %f %f\n' %(vn[i,0],vn[i,1],vn[i,2]))
        for i in range(f.shape[0]):
            fid.write('f %d %d %d\n' %(f[i,0]+1,f[i,1]+1,f[i,2]+1))
    print('save %s success' %(name))

if __name__=='__main__':
    filepath="/mnt/env/data/n.obj"
    savepath = "/mnt/env/data/n1.obj"
    v,f=load_obj(filepath)
    vn=cal_vn(v, f)
    save_obj(v,vn,f,savepath)
