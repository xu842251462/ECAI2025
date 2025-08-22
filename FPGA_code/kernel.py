from pynq import (allocate, Overlay)
import numpy as np

ol = Overlay('acceleration_kernel.bit')

# dma = ol.axi_dma_0
# mmult_ip = ol.hls_inst

#for 128 size
dma = ol.dma
mmult_ip = ol.accel

DIM = 128
in_buffer = allocate(shape=(2, DIM, DIM), dtype=np.float32, cacheable=False)
out_buffer = allocate(shape=(DIM, DIM), dtype=np.float32, cacheable=False)

CTRL_REG = 0x00
AP_START = (1 << 0)  # bit 0
AUTO_RESTART = (1 << 7)  # bit 7
mmult_ip.register_map.k = DIM
mmult_ip.register_map.m = DIM
mmult_ip.register_map.n = DIM

def run_kernel(a1, a2):
    # for hidden layer 8
#     a1_update = a1[:, :8]
#     a2_update = np.pad(a2, ((0, 43), (0, 0)), mode='constant', constant_values=0)[:64].reshape(8, 8)

    # for 128
    if a1.shape[1] > 128:
        # Truncate the extra columns if there are more than 128 columns
        a1_update = a1[:, :128]
    else:
        # Pad with zeros if there are fewer than 128 columns
        a1_update = np.pad(a1, ((0, 0), (0, 128 - a1.shape[1])), mode='constant', constant_values=0)

    if a2.shape[0] > 128:
        # Truncate the rows to 128
        a2_truncated = a2[:128, :]
    else:
        a2_truncated = a2  # No truncation needed
        
    a2_update = np.pad(a2_truncated, ((0, 0), (0, 128 - a2_truncated.shape[1])), mode='constant', constant_values=0)

#for 64
#     if a1.shape[1] > 64:
#         # Truncate the extra columns if there are more than 64 columns
#         a1_update = a1[:, :64]
#     else:
#         # Pad with zeros if there are fewer than 64 columns
#         a1_update = np.pad(a1, ((0, 0), (0, 64 - a1.shape[1])), mode='constant', constant_values=0)

#     if a2.shape[0] > 64:
#         # Truncate the rows to 64
#         a2_truncated = a2[:64, :]
#     else:
#         a2_truncated = a2  # No truncation needed

#     a2_update = np.pad(a2_truncated, ((0, 0), (0, 64 - a2_truncated.shape[1])), mode='constant', constant_values=0)

#for 32
#     if a1.shape[1] > 32:
#         # Truncate the extra columns if there are more than 32 columns
#         a1_update = a1[:, :32]
#     else:
#         # Pad with zeros if there are fewer than 32 columns
#         a1_update = np.pad(a1, ((0, 0), (0, 32 - a1.shape[1])), mode='constant', constant_values=0)

#     if a2.shape[0] > 32:
#         # Truncate the rows to 32
#         a2_truncated = a2[:32, :]
#     else:
#         a2_truncated = a2  # No truncation needed

#     a2_update = np.pad(a2_truncated, ((0, 0), (0, 32 - a2_truncated.shape[1])), mode='constant', constant_values=0)

#for16
#     if a1.shape[1] > 16:
#         # Truncate the extra columns if there are more than 16 columns
#         a1_update = a1[:, :16]
#     else:
#         # Pad with zeros if there are fewer than 16 columns
#         a1_update = np.pad(a1, ((0, 0), (0, 16 - a1.shape[1])), mode='constant', constant_values=0)

#     if a2.shape[0] > 16:
#         # Truncate the rows to 16
#         a2_truncated = a2[:16, :]
#     else:
#         a2_truncated = a2  # No truncation needed

#     a2_update = np.pad(a2_truncated, ((0, 0), (0, 16 - a2_truncated.shape[1])), mode='constant', constant_values=0)



    in_buffer[:] = np.stack((a1_update, a2_update))
    dma.sendchannel.transfer(in_buffer)
    dma.recvchannel.transfer(out_buffer)
    mmult_ip.write(CTRL_REG, (AP_START | AUTO_RESTART))  # initialize the module
    dma.sendchannel.wait()
    dma.recvchannel.wait()
    output = out_buffer[0:DIM, 0:1]
    return output
