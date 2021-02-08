# Protein Structure Modeling 

To build protein structure model, We used Rosetta’s protein folding and energy minimization protocols with customized constraints. The constraints were computed from our predictions of distance distributions (Cβ-Cβ, SCE-SCE, and backbone N-O) and angle distributions (backbone φ–ψ and the three residue-pair orientation angles) by normalizing the predicted values with predicted reference distributions.

## Installation
- To use trRosetta_modified.py and other python scripts, please install PyRosetta3: http://www.pyrosetta.org/dow/pyrosetta3-download
- The trRosetta_modified.py and other python scripts were modified from trRosetta modeling scripts downloaded from https://yanglab.nankai.edu.cn/trRosetta/download/

## Command for modeling
```
python trRosetta.py [json file]
```
In the json format file, the input npz file, target sequence file, output file and parameters are specified.

## json file format:
```
{
    "INI_MOD": "", 
    "phi": {
        "weight": 0.5
    }, 
    "psi": {
        "weight": 0.5
    }, 
    "dist": {
        "pcut": 0.5, #probability cut-off
        "weight": 1.0
    }, 
    "hbond": {
        "pcut": 0.5, 
        "weight": 0.8
    }, 
    "SEQ": "INPUT.seq", #fasta format sequence file
    "keys": [ #do not change
        "dist", 
        "phi", 
        "psi", 
        "omega", 
        "theta", 
        "orientation_phi", 
        "hbond", 
        "sidechain_center"
    ], 
    "orientation_phi": { 
        "pcut": 0.5, 
        "weight": 5.0
    }, 
    "OUT_MOD": "output.pdb", #output file name
    "mode": 3, #do not change. random folding path mode
    "NPZ": "INPUT.npz", #input npz file
    "theta": {
        "pcut": 0.5, 
        "weight": 5.0
    }, 
    "sidechain_center": {
        "pcut": 0.5, 
        "weight": 0.8
    }, 
    "omega": {
        "pcut": 0.5, 
        "weight": 5.0
    }
}
```

The trRosetta.py and other python scripts were modified from trRosetta modeling scripts downloaded from https://yanglab.nankai.edu.cn/trRosetta/download/

The following section shows details where we modified from the original code.

## arguments.py
### L6-8 need to be modified:
```
parser.add_argument("NPZ", type=str, help="input distograms and anglegrams (NN predictions)")
parser.add_argument("FASTA", type=str, help="input sequence")
parser.add_argument("OUT", type=str, help="output model (in PDB format)")
```
to
```
parser.add_argument("JSON", type=str, help="Input file (in JSON format)")
```

## trRosetta.py
### L44 Add:
```
##input json file modified
    with open(args.JSON) as jsonfile:
        input_npz = json.load(jsonfile)

    OUT = input_npz['OUT_MOD']
```

### L53-58 modify:
```
# read and process restraints & sequence
    npz = np.load(args.NPZ)
    seq = read_fasta(args.FASTA)
    L = len(seq)
    params['seq'] = seq
    rst = gen_rst(npz,tmpdir,params)
```
to
```
    seq = read_fasta(input_npz['SEQ'])
    L = len(seq)
    params['seq'] = seq
    rst = gen_rst_multi(tmpdir,params,input_npz,args.ave_ene)
```
### L93 for minimization by cartesian coordinate:
```min_mover_cart.cartesian(True)```

### L93 for minimization by torsion angle
```min_mover_cart.cartesian(False)```

### L96 Add:
```
mode = input_npz['mode']
```

### L117, L140, L156 modify
```args.mode```
to
```mode```

### L164- Add
```
    elif mode == 3:

        # short
        print('##short')
        add_rst(pose, rst, 1, 10, params)
        repeat_mover.apply(pose)
        min_mover_cart.apply(pose)
        remove_clash(sf_vdw, min_mover1, pose)

        print('##Foldinga by Random Path')
        Etbl=sum_ene_region(rst,len(seq),10)
        step=1
        random.shuffle(Etbl)
        for i,j,e in Etbl:
            print('step',step,i,j,e)
            chk = 0
            if e < 0.00 :
                chk=add_rst_region(pose, rst,i,j,10, params)
            if chk != 0:
                repeat_mover.apply(pose)
                min_mover_cart.apply(pose)
                remove_clash(sf_vdw, min_mover1, pose)
                step=step+1
```

## utilts_ros.py
### Add new function gen_rst_multi(). This function generate customized restrain files from npz file.
```
def gen_rst_multi(tmpdir, params,npz_list,aflag):

    #load npz files
    NPZ = np.load(npz_list['NPZ'])

    ########################################################
    # assign parameters
    ########################################################
    PCUT  = 0.05 #params['PCUT']
    PCUT1 = params['PCUT1']
    EBASE = params['EBASE']
    EREP  = params['EREP']
    DREP  = params['DREP']
    PREP  = params['PREP']
    SIGD  = params['SIGD']
    SIGM  = params['SIGM']
    MEFF  = params['MEFF']
    DCUT  = params['DCUT']
    ALPHA = params['ALPHA']

    DSTEP = params['DSTEP']
    ASTEP = np.deg2rad(params['ASTEP'])
    BB_ASTEP = np.deg2rad(10.00) ##backbone phi, psi angle

    seq = params['seq']

    # dictionary to store ALL restraints
    rst = {'dist' : [], 'omega' : [], 'theta' : [], 'phi' : [], 'rep' : [], 'hbond' : [],'sce' : [],'bbphi' :[] ,'bbpsi': [], 'minene': []}


    ########################################################
    # Cb-Cb dist:20 bins 0:0~4A, 1:4~4.5A, 2:4.5~5A , 19:18~inf
    ########################################################

    key = 'dist'
    rkey = 'dist_ref_pred'
    if key in npz_list['keys']:
        PCUT = npz_list[key]['pcut']
        weight = npz_list[key]['weight']
        dist = NPZ[key].transpose(1,2,0)
        refp = NPZ[rkey].transpose(1,2,0)
        print(key,dist.shape);
        print(rkey,refp.shape);
        nres = dist.shape[0]
        #1bins[0-4]:0~4.0A
        rst['minene'] = np.zeros((nres,nres))
        bins = [4.25,4.75,5.25,5.75,6.25,6.75,7.25,7.75,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5]
        prob = np.sum(dist[:,:,1:19], axis=-1) #18bins:4.0~18A, ignore 1st and final bins
        CbCb_prob = np.sum(dist[:,:,1:19], axis=-1) #18bins:4.0~18A
        #normarized by reference prob
        attr = -np.log((dist[:,:,1:19]+MEFF)/(refp[:,:,1:19]+MEFF))+EBASE
        repul = np.maximum(attr[:,:,0],np.zeros((nres,nres)))[:,:,None]+np.array(EREP)[None,None,:]
        dist = np.concatenate([repul,attr], axis=-1)
        bins = np.concatenate([DREP,bins]) #3bins[0.0,2.0,3.5A] + 18bins[4.0~20A]
        i,j = np.where(prob>PCUT)
        prob = prob[i,j]
        nbins = 21
        step = 0.5
        for a,b,p in zip(i,j,prob):
            if b>a:
                name=tmpdir.name+"/%d.%d.dist.txt"%(a+1,b+1)
                with open(name, "w") as f:
                    f.write('x_axis'+'\t%.3f'*nbins%tuple(bins)+'\n')
                    f.write('y_axis'+'\t%.3f'*nbins%tuple(dist[a,b])+'\n')
                    f.close()
                rst_line = 'AtomPair %s %d %s %d SPLINE TAG %s 1.0 %.3f %.5f'%('CB',a+1,'CB',b+1,name,weight,step)
                rst['dist'].append([a,b,p,rst_line])
                rst['minene'][a,b]=dist[a,b].min()
                #print('AtomPair %d %d min=%f'%(a,b,rst['minene'][a,b]))
        print("Cb-Cb restraints:  %d"%(len(rst['dist'])))


    ########################################################
    # Side-chain center dist:38 bins 0:0~2A, 1:2~2.5A, 2:2.5~3A , 37:20~inf
    ########################################################
    key = 'sidechain_center'
    rkey = 'sidechain_ref_pred'
    if key in npz_list['keys']:
        PCUT = npz_list[key]['pcut']
        weight = npz_list[key]['weight']
        dist = NPZ[key].transpose(1,2,0)
        refp = NPZ[rkey].transpose(1,2,0)
        print("sidechain_center",dist.shape);
        print("sidechain_ref",refp.shape);
        nres = dist.shape[0]
        #5bins[0-4]:0~4.0A
        bins = np.array([4.25+DSTEP*i for i in range(32)])
        prob = np.sum(dist[:,:,5:37], axis=-1) #32bins:4.0~20A, ignore 0,1,2,3,4 and final bins
        #normarized by reference prob
        attr = -np.log((dist[:,:,5:37]+MEFF)/(refp[:,:,5:37]+MEFF))+EBASE
        repul = np.maximum(attr[:,:,0],np.zeros((nres,nres)))[:,:,None]+np.array(EREP)[None,None,:]
        dist = np.concatenate([repul,attr], axis=-1)
        bins = np.concatenate([DREP,bins]) #3bins[0.0,2.0,3.5A] + 32bins[3.0~20A]
        i,j = np.where(prob>PCUT)
        prob = prob[i,j]
        nbins = 35
        step = 0.5
        for a,b,p in zip(i,j,prob):
            if b>a :
                name=tmpdir.name+"/%d.%d.SCE.txt"%(a+1,b+1)
                with open(name, "w") as f:
                    f.write('x_axis'+'\t%.3f'*nbins%tuple(bins)+'\n')
                    f.write('y_axis'+'\t%.3f'*nbins%tuple(dist[a,b])+'\n')
                    f.close()
                rst_line = 'AtomPair %s %d %s %d SPLINE TAG %s 1.0 %.3f %.5f'%('CEN',a+1,'CEN',b+1,name,weight,step)
                rst['sce'].append([a,b,p,rst_line])
        print("Sce restraints:  %d"%(len(rst['sce'])))


        ########################################################
        # Hbonds dist: 0..20A 38bins 0:0~2A, 1:2~2.5A, 2:2.5~3A , 37:20~inf
        ########################################################
    key = 'hbond'
    rkey = 'hbond_ref_pred'
    #NO
    if key in npz_list['keys']:
        PCUT = npz_list[key]['pcut']
        weight = npz_list[key]['weight']
        dist = NPZ[key].transpose(1,2,0)
        refp = NPZ[rkey].transpose(1,2,0)
        print("hbond",dist.shape);
        print("hbond_ref_pred",refp.shape);
        nres = dist.shape[0]
        H_DREP = [0,2.25]
        H_EREP = [10.0,0.5]
        bins = np.array([2.75+DSTEP*i for i in range(35)])
        #prob = np.sum(dist[:,:,2:7], axis=-1) #only 2.5~5.0
        prob = np.sum(dist[:,:,2:37], axis=-1) #2.5~20.0
        #normarized by reference prob
        attr = -np.log((dist[:,:,2:37]+MEFF)/(refp[:,:,2:37]+MEFF))+EBASE
        repul = np.maximum(attr[:,:,0],np.zeros((nres,nres)))[:,:,None]+np.array(H_EREP)[None,None,:]
        dist = np.concatenate([repul,attr], axis=-1)
        bins = np.concatenate([H_DREP,bins]) #0,2.25

        i,j = np.where(prob>PCUT)
        prob = prob[i,j]
        nbins = 37
        step = 0.5
        for a,b,p in zip(i,j,prob):
            if b>a+2 or a>b+2:
                name=tmpdir.name+"/%d.%d.HNO.txt"%(a+1,b+1)
                with open(name, "w") as f:
                    f.write('x_axis'+'\t%.3f'*nbins%tuple(bins)+'\n')
                    f.write('y_axis'+'\t%.3f'*nbins%tuple(dist[a,b])+'\n')
                    f.close()
                rst_line = 'AtomPair %s %d %s %d SPLINE TAG %s 1.0 %.3f %.5f'%('N',a+1,'O',b+1,name,weight,step)
                rst['hbond'].append([a,b,p,rst_line])
        print("Hbonds restraints:  %d"%(len(rst['hbond'])))


    key = 'omega'
    rkey = 'omega_ref_pred'
    if key in npz_list['keys']:
        PCUT = npz_list[key]['pcut']
        weight = npz_list[key]['weight']
        omega = NPZ[key].transpose(1,2,0)
        refp = NPZ[rkey].transpose(1,2,0)
        print("omega",omega.shape);
        print("omega_ref",refp.shape);
        nres = omega.shape[0]
        nbins = omega.shape[2]-1+4 #ignore last bin, add 2+2 bins
        bins = np.linspace(-np.pi-1.5*ASTEP, np.pi+1.5*ASTEP, nbins)
        prob = np.sum(omega[:,:,0:24], axis=-1) #0-23:contact, 24:no-contact
        i,j = np.where(prob>PCUT)
        prob = prob[i,j]
        omega = -np.log((omega[:,:,0:24]+MEFF)/(refp[:,:,0:24]+MEFF))
        omega = np.concatenate([omega[:,:,-2:],omega[:,:,0:],omega[:,:,0:2]],axis=-1)
        for a,b,p in zip(i,j,prob):
            #if b>a:
            if b>a and CbCb_prob[a,b]>PCUT:
                #print("Lastbin= %f P= %f"%(omega[a,b,-1],p))
                name=tmpdir.name+"/%d.%d_omega.txt"%(a+1,b+1)
                with open(name, "w") as f:
                    f.write('x_axis'+'\t%.5f'*nbins%tuple(bins)+'\n')
                    f.write('y_axis'+'\t%.5f'*nbins%tuple(omega[a,b])+'\n')
                    f.close()
                rst_line = 'Dihedral CA %d CB %d CB %d CA %d SPLINE TAG %s 1.0 %.3f %.5f'%(a+1,a+1,b+1,b+1,name,weight,ASTEP)
                rst['omega'].append([a,b,p,rst_line])
        print("omega restraints: %d"%(len(rst['omega'])))


        ########################################################
        # theta: -pi..pi
        ########################################################
    key = 'theta'
    rkey = 'theta_ref_pred'
    if key in npz_list['keys']:
        PCUT = npz_list[key]['pcut']
        weight = npz_list[key]['weight']
        theta = NPZ[key].transpose(1,2,0)
        refp = NPZ[rkey].transpose(1,2,0)
        print("theta",theta.shape);
        print("theta_ref",refp.shape);
        nres = theta.shape[0]
        nbins = theta.shape[2]-1+4 #ignore last bin, add 2+2 bins
        bins = np.linspace(-np.pi-1.5*ASTEP, np.pi+1.5*ASTEP, nbins)
        prob = np.sum(theta[:,:,0:24], axis=-1) #0-23:contact, 24:no-contact

        i,j = np.where(prob>PCUT)
        prob = prob[i,j]
        theta = -np.log((theta[:,:,0:24]+MEFF)/(refp[:,:,0:24]+MEFF))
        theta = np.concatenate([theta[:,:,22:24],theta[:,:,0:],theta[:,:,0:2]],axis=-1)
        for a,b,p in zip(i,j,prob):
            #if b!=a:
            if b!=a and CbCb_prob[a,b]>PCUT:
                name=tmpdir.name+"/%d.%d_theta.txt"%(a+1,b+1)
                with open(name, "w") as f:
                    f.write('x_axis'+'\t%.3f'*nbins%tuple(bins)+'\n')
                    f.write('y_axis'+'\t%.3f'*nbins%tuple(theta[a,b])+'\n')
                    f.close()
                rst_line = 'Dihedral N %d CA %d CB %d CB %d SPLINE TAG %s 1.0 %.3f %.5f'%(a+1,a+1,a+1,b+1,name,weight,ASTEP)
                rst['theta'].append([a,b,p,rst_line])
        print("theta restraints: %d"%(len(rst['theta'])))
    key = 'orientation_phi'
    rkey = 'ori_phi_ref_pred'
    if key in npz_list['keys']:
        PCUT = npz_list[key]['pcut']
        weight = npz_list[key]['weight']
        phi = NPZ[key].transpose(1,2,0)
        refp = NPZ[rkey].transpose(1,2,0)

        print("phi",phi.shape);
        print("phi_ref",refp.shape);
        nbins = phi.shape[2]-1+4 #ignore last bin, use 0~11bin
        bins = np.linspace(-1.5*ASTEP, np.pi+1.5*ASTEP, nbins)
        #prob = np.sum(phi[:,:,1:], axis=-1)
        prob = np.sum(phi[:,:,0:12], axis=-1) #0-11: contact, 12:no-contact
        i,j = np.where(prob>PCUT)
        prob = prob[i,j]
        phi = -np.log((phi[:,:,0:12]+MEFF)/(refp[:,:,0:12]+MEFF))

        #[1,0] + [0,1,....11]+[11,10]
        phi = np.concatenate([np.flip(phi[:,:,0:2],axis=-1),phi[:,:,0:12],np.flip(phi[:,:,10:12],axis=-1)], axis=-1)
        for a,b,p in zip(i,j,prob):
            #if b!=a:
            if b!=a and CbCb_prob[a,b]>PCUT:
                name=tmpdir.name+"/%d.%d_phi.txt"%(a+1,b+1)
                with open(name, "w") as f:
                    f.write('x_axis'+'\t%.3f'*nbins%tuple(bins)+'\n')
                    f.write('y_axis'+'\t%.3f'*nbins%tuple(phi[a,b])+'\n')
                    f.close()
                rst_line = 'Angle CA %d CB %d CB %d SPLINE TAG %s 1.0 %.3f %.5f'%(a+1,a+1,b+1,name,weight,ASTEP)
                rst['phi'].append([a,b,p,rst_line])

        print("phi restraints:   %d"%(len(rst['phi'])))

    #Backbone phi -pi~pi, 36 bin
    key = 'phi'
    rkey = 'phi_ref_pred'
    if key in npz_list['keys']:
        weight = npz_list[key]['weight']
        phi = NPZ[key].transpose(1,0)
        refp = NPZ[rkey].transpose(1,0)

        print("bbphi",phi.shape);
        print("bbphi_ref",refp.shape);

        nbins = phi.shape[1]+4 #use 0~35bins + 4 bins
        bins = np.linspace(-np.pi-1.5*BB_ASTEP, np.pi+1.5*BB_ASTEP, nbins)
        phi = -np.log((phi[:,0:36]+MEFF)/(refp[:,0:36]+MEFF))

        #[34,35] + [0,1,....35]+[0,1]
        phi = np.concatenate([phi[:,34:36],phi[:,0:36],phi[:,0:2]],axis=-1)
        for a in range(1,phi.shape[0]): #ignore 1st residue
            name=tmpdir.name+"/%d_bbphi.txt"%(a+1)
            with open(name, "w") as f:
                f.write('x_axis'+'\t%.3f'*nbins%tuple(bins)+'\n')
                f.write('y_axis'+'\t%.3f'*nbins%tuple(phi[a])+'\n')
                f.close()

            rst_line = 'Dihedral C %d N %d CA %d C %d SPLINE TAG %s 1.0 %.3f %.5f'%(a,a+1,a+1,a+1,name,weight,BB_ASTEP)
            rst['bbphi'].append([a,rst_line])
        print("bbphi restraints:   %d"%(len(rst['bbphi'])))

     #Backbone phi -pi~pi, 36 bin
    key = 'psi'
    rkey = 'psi_ref_pred'
    if key in npz_list['keys']:
        weight = npz_list[key]['weight']
        psi = NPZ[key].transpose(1,0)
        refp = NPZ[rkey].transpose(1,0)

        print("bbpsi",phi.shape);
        print("bbpsi_ref",refp.shape);

        nbins = psi.shape[1]+4 #use 0~35bins + 4 bins
        bins = np.linspace(-np.pi-1.5*BB_ASTEP, np.pi+1.5*BB_ASTEP, nbins)
        psi = -np.log((psi[:,0:36]+MEFF)/(refp[:,0:36]+MEFF))

        #[34,35] + [0,1,....35]+[0,1]
        psi = np.concatenate([psi[:,34:36],psi[:,0:36],psi[:,0:2]],axis=-1)
        #for a,b,p in zip(i,j,prob):
        for a in range(0,psi.shape[0]-1): #ignore last residue
            name=tmpdir.name+"/%d_bbpsi.txt"%(a+1)
            with open(name, "w") as f:
                f.write('x_axis'+'\t%.3f'*nbins%tuple(bins)+'\n')
                f.write('y_axis'+'\t%.3f'*nbins%tuple(phi[a])+'\n')
                f.close()

            rst_line = 'Dihedral N %d CA %d C %d N %d SPLINE TAG %s 1.0 %.3f %.5f'%(a+1,a+1,a+1,a+2,name,weight,BB_ASTEP)
            rst['bbpsi'].append([a,rst_line])
        print("bbpsi restraints:   %d"%(len(rst['bbpsi'])))
```

### L210 Change add_rst() as
```
def add_rst(pose, rst, sep1, sep2, params, nogly=False):

    #ignore pcut
    seq = params['seq']

    array=[]

    #print pose.energies().residue_total_energies(3)[cen_hb]
    if nogly==True:
        array += [line for a,b,p,line in rst['dist'] if abs(a-b)>=sep1 and abs(a-b)<sep2 and seq[a]!='G' and seq[b]!='G' ]
        print("#CbCb= ",len(array))
        array += [line for a,b,p,line in rst['hbond'] if abs(a-b)> 3 and abs(a-b)>=sep1 and abs(a-b)<sep2 ]
        print("#CbCb+hbond= ",len(array))
        if params['USE_ORIENT'] == True:
            array += [line for a,b,p,line in rst['omega'] if abs(a-b)>=sep1 and abs(a-b)<sep2 and seq[a]!='G' and seq[b]!='G' ]
            array += [line for a,b,p,line in rst['theta'] if abs(a-b)>=sep1 and abs(a-b)<sep2 and seq[a]!='G' and seq[b]!='G' ]
            array += [line for a,b,p,line in rst['phi'] if abs(a-b)>=sep1 and abs(a-b)<sep2 and seq[a]!='G' and seq[b]!='G' ]
    else:
        array += [line for a,b,p,line in rst['dist'] if abs(a-b)>=sep1 and abs(a-b)<sep2 ]
        array += [line for a,b,p,line in rst['sce'] if abs(a-b)>=sep1 and abs(a-b)<sep2 ]
        print("#CbCb+SCE= ",len(array))
        array += [line for a,b,p,line in rst['hbond'] if abs(a-b)> 3 and abs(a-b)>=sep1 and abs(a-b)<sep2 ]
        if params['USE_ORIENT'] == True:
            array += [line for a,b,p,line in rst['omega'] if abs(a-b)>=sep1 and abs(a-b)<sep2 ]
            array += [line for a,b,p,line in rst['theta'] if abs(a-b)>=sep1 and abs(a-b)<sep2 ]
            array += [line for a,b,p,line in rst['phi'] if abs(a-b)>=sep1 and abs(a-b)<sep2 ]
        print("#CbCb+SCE+Hbonds+Orient= ",len(array))


    ##bb-phi,psi
    array += [line for a, line in rst['bbpsi']]
    array += [line for a, line in rst['bbphi']]

    print("#CbCb+SCE+Hbonds+Orient+BBdangle= ",len(array))

    if len(array) < 1:
        return

    random.shuffle(array)

    # save to file
    tmpname = params['TDIR']+'/minimize.cst'
    #tmpname = './minimize.cst'
    with open(tmpname,'w') as f:
        for line in array:
            f.write(line+'\n')
        f.close()

    print("#Wrote",tmpname)
    # add to pose
    constraints = rosetta.protocols.constraint_movers.ConstraintSetMover()
    constraints.constraint_file(tmpname)
    constraints.add_constraints(True)
    constraints.apply(pose)

    print("#Remove ",tmpname)
    os.remove(tmpname)
```

### L250 add three functions, sum_ene_region(),  fold_order() and add_rst_region()  as
```
def sum_ene_region(rst,L,Lunit):
    Nunit=(int)(L/Lunit)+1
    print(L,Lunit,'Nunit',Nunit)
    Eregion=np.zeros((Nunit,Nunit))

    array_dia=[]
    array=[]

    Ntmp=0
    for i in range(0,Nunit):
        for j in range(i,Nunit):
            Eregion[i,j]=rst['minene'][i*Nunit:(i+1)*Nunit,j*Nunit:(j+1)*Nunit].sum()
            if Eregion[i,j] < -50.0 :
                print('Eregion',i,j,Eregion[i,j])
                if abs(i-j) < 2 :
                    array_dia.append([i,j,Eregion[i,j]])
                else :
                    array.append([i,j,Eregion[i,j]])
                    Ntmp=Ntmp+1

    random.shuffle(array)
    print('TotalUnit=%d'%(Ntmp))
    if Ntmp >Nunit :
        array=array[0:Nunit]

    print(array)

    return(array)


def fold_order(pose,rst,params,Etbl,L):
    print('##Folding')
    step=0
    random.shuffle(Etbl)
    for i,j,e in Etbl:
        print(i,j,e)
        add_rst_region(pose, rst,i,j,L, params)
        repeat_mover.apply(pose)
        min_mover_cart.apply(pose)
        remove_clash(sf_vdw, min_mover1, pose)
        pose.dump_pdb(step+"short.pdb")
        step=step+1

def add_rst_region(pose, rst, rg1,rg2,L, params):

    #ignore pcut
    seq = params['seq']
    pos11=rg1*L
    pos12=rg1*L+L
    pos21=rg2*L
    pos22=rg2*L+L

    array=[]

    #print pose.energies().residue_total_energies(3)[cen_hb]
    array += [line for a,b,p,line in rst['dist'] if (a>=pos11 and a<pos12) and (b>=pos21 and b<pos22) ]
    array += [line for a,b,p,line in rst['sce'] if (a>=pos11 and a<pos12) and (b>=pos21 and b<pos22) ]
    print("#CbCb+SCE= ",len(array))
    array += [line for a,b,p,line in rst['hbond'] if ((a>=pos11 and a<pos12) and (b>=pos21 and b<pos22)) or ((b>=pos11 and b<pos12) and (a>=pos21 and a<pos22))]
    if params['USE_ORIENT'] == True:
        array += [line for a,b,p,line in rst['omega'] if ((a>=pos11 and a<pos12) and (b>=pos21 and b<pos22)) or  ((b>=pos11 and b<pos12) and (a>=pos21 and a<pos22)) ]
        array += [line for a,b,p,line in rst['theta'] if ((a>=pos11 and a<pos12) and (b>=pos21 and b<pos22)) or  ((b>=pos11 and b<pos12) and (a>=pos21 and a<pos22)) ]
        array += [line for a,b,p,line in rst['phi'] if ((a>=pos11 and a<pos12) and (b>=pos21 and b<pos22)) or  ((b>=pos11 and b<pos12) and (a>=pos21 and a<pos22)) ]

    if len(array) < 1:
        return 0
    random.shuffle(array)


    # save to file
    tmpname = params['TDIR']+'/minimize.cst'
    with open(tmpname,'w') as f:
        for line in array:
            f.write(line+'\n')
        f.close()

    print("#Wrote",tmpname)
    # add to pose
    constraints = rosetta.protocols.constraint_movers.ConstraintSetMover()
    constraints.constraint_file(tmpname)
    constraints.add_constraints(True)
    constraints.apply(pose)
    print("#Remove ",tmpname)
    os.remove(tmpname)

```
