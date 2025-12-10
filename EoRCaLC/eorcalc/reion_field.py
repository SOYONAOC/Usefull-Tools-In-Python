# Import Standard Modules
import numpy as np
import torch
import pandas as pd
import logging
import os

# Import self module relesed to PyPI(pip install massfunc bubblebarrier xxiop)
from xxiop.op import OpticalDepth
from .special import load_binary_data,TopHat_filter,xHII_field_update
from .ioninti_gpu import Ion

### create logger


def reionization_calculator(fesc=0.2,A2byA1=1.0,kMpc_trans=1e6,alpha=0.0,beta=0.0,label = 'MH',DIM=256,box_length=800,save_on=True):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    ### create logger
    log_file = f'logfile/reionization_{label}.log'
    os.makedirs('logfile', exist_ok=True)
    # 使用filemode='w'覆盖模式，不需要手动删除
    logging.basicConfig(
        filename=log_file, 
        level=logging.INFO,
        filemode='w'  # 'w'表示覆盖写入，'a'表示追加
    )
    z_value = np.load('redshift_list.npy')
    z_value = z_value[::-1]  # reverse the order to have higher z first
    dtype = torch.float32
    xHII_field = torch.zeros((DIM,DIM,DIM), dtype=dtype, device=device)
    nrec_field = torch.zeros((DIM,DIM,DIM), dtype=dtype, device=device)
    logging.info(f'Start reionization calculation with fesc={fesc}, kMpc_trans={kMpc_trans}')
    ionf = []
    
    ### main loop
    for i,z in enumerate(z_value):
        dz = z_value[i+1] - z_value[i] if i < len(z_value) - 1 else z_value[i] - z_value[i-1]
        ion = Ion(fesc=fesc,z=z,A2byA1=A2byA1,ktrans=kMpc_trans,alpha=alpha,beta=beta)
        read_path = f'kinf/updated_smoothed_deltax_z{z:06.2f}_{int(DIM)}_{int(box_length)}Mpc'
        # read_path = f'df/k{kMpc_trans:.0f}/updated_smoothed_deltax_z{z:06.2f}_{int(DIM)}_{int(box_length)}Mpc'
        ### load the density field
        delta_field_cpu = load_binary_data(read_path, DIM=DIM).astype(np.float32)
        # logging.info(f'z={z:.2f}, density std: {np.std(delta_field_cpu):.4f}')
        delta_field = torch.as_tensor(delta_field_cpu, dtype=dtype, device=device)
        del delta_field_cpu

        ### main smoothing loop
        delta_field_ffted = torch.fft.rfftn(delta_field,norm="forward")
        nrec_field_ffted = torch.fft.rfftn(nrec_field,norm="forward")
        # del delta_field

        rs = np.logspace(np.log10(box_length/DIM), np.log10(50), 50)
        
        mean_nxi = ion.nxi_st(z)
        rs=rs[::-1]
        ### Smooth the density field and recombination field at different scales
        for j,r in enumerate(rs):
            deltav_smoothed = TopHat_filter(delta_field_ffted,R=r,DIM=DIM,box_length=box_length)
            mv=ion.cosmo.rhom*4*np.pi/3*r**3
            mean_nion = ion.nion_st(z,mv)
            ### source smoothed
            source_smoothed = ion.nion_interp(mv,deltav_smoothed)
            source_ratio = mean_nion / torch.mean(source_smoothed)
            ### IGM smoothed
            nrec_smoothed = TopHat_filter(nrec_field_ffted,R=r,DIM=DIM,box_length=box_length)
            igm_smoothed = (1+nrec_smoothed)*ion.n_HI(deltav_smoothed)*(1-ion.cosmo.fcoll_st(z,ion.M_J,ion.M_min,5000))
            # logging.info(f'z={z:.2f}, R={r:.2f} Mpc, nhi mean: {torch.mean(ion.n_HI(deltav_smoothed)):.4e},fcoll: {ion.cosmo.fcoll_st(z,ion.M_J,ion.M_min,5000):.4f}')
            # logging.info(f'z={z:.2f}, nrec_smoothed mean: {torch.mean(nrec_smoothed):.4e}, mean IGM: {torch.mean(igm_smoothed):.4e}')
            ### minihalo smoothed
            minihalo_smoothed = ion.nxi_interp(mv,deltav_smoothed)
            minihalo_ratio = mean_nxi / torch.mean(minihalo_smoothed)
            # logging.info(f'z={z:.2f}, minihalo_ratio={minihalo_ratio:.4f}, source_ratio={source_ratio:.4f}, R={r:.2f} Mpc')
            # logging.info(f'z={z:.2f}, source: {torch.mean(source_smoothed):.4e}, minihalo: {torch.mean(minihalo_smoothed):.4e}')
            # logging.info(f'z={z:.2f}, mean source {mean_nion:.4e}, mean minihalo: {mean_nxi:.4e}')
            del deltav_smoothed,nrec_smoothed
            mask = source_ratio*source_smoothed > ( igm_smoothed + minihalo_ratio*minihalo_smoothed)
            # logging.info(f'z={z:.2f}, mask sum: {cp.sum(mask):.4e}')
            xHII_field[mask] = 1.0
            del mask
            if j == 49:
                partial_eff = source_ratio*source_smoothed / ( igm_smoothed + minihalo_ratio*minihalo_smoothed)
                # logging.info(f'z={z:.2f}, source: {torch.mean(source_smoothed):.4e}, igm: {torch.mean(igm_smoothed):.4e}, minihalo: {torch.mean(minihalo_smoothed):.4e}, source_ratio: {source_ratio:.4f}, minihalo_ratio: {minihalo_ratio:.4f}')
                partial_eff = partial_eff.to(dtype)
                mean_fraction = torch.mean(xHII_field)
                xHII_field = xHII_field_update(xHII_field,partial_eff)
                # logging.info(f'z={z:.2f}, partial: {torch.mean(partial_eff):.4f}, device: {partial_eff.device}')
                logging.info(f'z={z:.2f}, ionization fraction after partial update: {mean_fraction*100:.4f} %')
                del partial_eff
            del source_smoothed,igm_smoothed,minihalo_smoothed

        del delta_field_ffted,nrec_field_ffted

        os.makedirs(f'reionf/{label}', exist_ok=True)
        file_save_path = f'reionf/{label}/rf_{z:.2f}.npy'
        if z <= 30.0:
            nrec_field += ion.dnrec_dz_path(delta_field,xHII_field) * dz
        del delta_field
        ionization_fraction_gpu = torch.mean(xHII_field)
        ionization_fraction = float(ionization_fraction_gpu)
        logging.info(f"z = : {z:.2f} ({ionization_fraction*100:.2f}%)")
        if save_on:
            torch.save(xHII_field, file_save_path)
        ionf.append((z, ionization_fraction))

    final_z_values = np.array([item[0] for item in ionf])
    final_fractions = np.array([item[1] for item in ionf])

    # 创建 DataFrame 并保存
    output_df = pd.DataFrame({
        'z': final_z_values,
        'ionf': final_fractions,
    })
    os.makedirs('csvfile', exist_ok=True)
    csv_save_path = f'csvfile/{label}.csv'
    output_df.to_csv(csv_save_path, index=False)
    # Calculate optical depth
    optical_depth = OpticalDepth(file_path=csv_save_path)
    depth20 = optical_depth.OpticalDepth(20)
    depth30 = optical_depth.OpticalDepth(30)

    print(f"Optical depth at z=20: {depth20:.4f}, at z=30: {depth30:.4f}")
    return depth30
