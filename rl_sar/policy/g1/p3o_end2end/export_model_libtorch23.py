import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/ubuntu/P3O-CBF/unitree_rl_lab/scripts/rsl_rl/paper_2508_07611')
import torch
import torch.nn as nn
from actor_critic_safe_perception import ActorCriticSafePerception, ObsTermSpec

class ManualGRU(nn.Module):
    def __init__(self, gru):
        super().__init__()
        H = gru.hidden_size
        I = gru.input_size
        W_ih = gru.weight_ih_l0
        W_hh = gru.weight_hh_l0
        b_ih = gru.bias_ih_l0
        b_hh = gru.bias_hh_l0
        # PyTorch GRU order: z, r, n
        self.Wz_i = nn.Linear(I, H, bias=False); self.Wz_i.weight.data = W_ih[:H]
        self.Wz_h = nn.Linear(H, H, bias=False); self.Wz_h.weight.data = W_hh[:H]
        self.bz_i = nn.Parameter(b_ih[:H]); self.bz_h = nn.Parameter(b_hh[:H])
        self.Wr_i = nn.Linear(I, H, bias=False); self.Wr_i.weight.data = W_ih[H:2*H]
        self.Wr_h = nn.Linear(H, H, bias=False); self.Wr_h.weight.data = W_hh[H:2*H]
        self.br_i = nn.Parameter(b_ih[H:2*H]); self.br_h = nn.Parameter(b_hh[H:2*H])
        self.Wn_i = nn.Linear(I, H, bias=False); self.Wn_i.weight.data = W_ih[2*H:]
        self.Wn_h = nn.Linear(H, H, bias=False); self.Wn_h.weight.data = W_hh[2*H:]
        self.bn_i = nn.Parameter(b_ih[2*H:]); self.bn_h = nn.Parameter(b_hh[2*H:])
        self.H = H

    def forward(self, x, h):
        # x: (B,T,I) h: (1,B,H)
        B,T,_ = x.shape
        h = h[0]
        outs = []
        for t in range(T):
            xt = x[:,t]
            z = torch.sigmoid(self.Wz_i(xt)+self.bz_i+self.Wz_h(h)+self.bz_h)
            r = torch.sigmoid(self.Wr_i(xt)+self.br_i+self.Wr_h(h)+self.br_h)
            n = torch.tanh(self.Wn_i(xt)+self.bn_i+r*(self.Wn_h(h)+self.bn_h))
            h = (1-z)*n+z*h
            outs.append(h.unsqueeze(1))
        return torch.cat(outs,1), h.unsqueeze(0)

class ManualEncoder(nn.Module):
    def __init__(self, orig):
        super().__init__()
        self.proprio_frame_encoder = orig.proprio_frame_encoder
        self.proprio_gru = ManualGRU(orig.proprio_gru)
        self.point_mlp = orig.point_mlp
        self.scan_frame_encoder = orig.scan_frame_encoder
        self.scan_gru = ManualGRU(orig.scan_gru)
        self.term_offsets = orig.term_offsets
        self.input_dim = orig.input_dim
        self.lidar_dim = orig.lidar_dim
        self.proprio_frame_dim = orig.proprio_frame_dim
        self.history_length = orig.history_length
        self.lidar_mode = orig.lidar_mode
        self.num_points = orig.num_points
        self.output_dim = orig.output_dim
        self._term_names = [s.name for s in orig.term_specs]
        self._term_dims = {s.name: s.dim for s in orig.term_specs}
        self._proprio_names = [s.name for s in orig.term_specs if s.name != 'lidar_points']

    def forward(self, observations):
        B = observations.shape[0]
        terms = {}
        for name in self._term_names:
            start, end, dim = self.term_offsets[name]
            terms[name] = observations[:, start:end].view(B, self.history_length, dim)
        scan_seq = terms['lidar_points']
        proprio_seq = torch.cat([terms[n] for n in self._proprio_names], dim=-1)
        B, T, D = proprio_seq.shape
        pf = self.proprio_frame_encoder(proprio_seq.reshape(B*T,D)).reshape(B,T,-1)
        _, ph = self.proprio_gru(pf, torch.zeros(1,B,self.proprio_gru.H))
        ph = ph[0]
        sr = scan_seq.reshape(B*T,self.num_points,3)
        sf = self.point_mlp(sr).max(dim=1).values
        sf = self.scan_frame_encoder(sf).view(B,T,-1)
        _, sh = self.scan_gru(sf, torch.zeros(1,B,self.scan_gru.H))
        sh = sh[0]
        sl = sf[:,-1,:]
        return torch.cat((ph,sh,sl),-1)

class ManualPolicy(nn.Module):
    def __init__(self, orig):
        super().__init__()
        self.actor_encoder = ManualEncoder(orig.actor_encoder)
        self.actor = orig.actor
    def act_inference(self, obs):
        return self.actor(self.actor_encoder(obs))

class Wrap(nn.Module):
    def __init__(self, p): super().__init__(); self.policy = p
    def forward(self, obs): return self.policy.act_inference(obs)

def main():
    ckpt = torch.load('/home/ubuntu/P3O-CBF/logs/P3O-END2END-001/2026-04-30_06-06-44/model_final.pt',
                       map_location='cpu', weights_only=False)
    sd = ckpt['policy_state_dict']
    actor_specs = [
        ObsTermSpec('base_ang_vel',3,5), ObsTermSpec('projected_gravity',3,5),
        ObsTermSpec('velocity_commands',3,5), ObsTermSpec('lidar_points',384,5),
        ObsTermSpec('joint_pos_rel',29,5), ObsTermSpec('joint_vel_rel',29,5), ObsTermSpec('last_action',29,5),
    ]
    critic_specs = [
        ObsTermSpec('base_lin_vel',3,5), ObsTermSpec('base_ang_vel',3,5),
        ObsTermSpec('projected_gravity',3,5), ObsTermSpec('velocity_commands',3,5),
        ObsTermSpec('lidar_points',384,5), ObsTermSpec('joint_pos_rel',29,5),
        ObsTermSpec('joint_vel_rel',29,5), ObsTermSpec('last_action',29,5),
    ]
    orig = ActorCriticSafePerception(
        actor_term_specs=actor_specs, critic_term_specs=critic_specs,
        num_actions=29, actor_hidden_dims=[256,128], critic_hidden_dims=[256,128],
        activation='elu', init_noise_std=1.0,
        proprio_hidden_dim=128, scan_hidden_dim=64, rnn_hidden_dim=64,
    )
    orig.load_state_dict(sd, strict=True)
    orig.eval()
    dummy = torch.zeros(1,2400)
    with torch.no_grad(): ref = orig.act_inference(dummy).numpy()
    print(f'Ref: {ref[0,:5]}')

    man = ManualPolicy(orig); man.eval()
    with torch.no_grad(): mout = man.act_inference(dummy).numpy()
    print(f'Man: {mout[0,:5]}')
    print(f'Diff: {abs(ref-mout).max():.10f}')

    w = Wrap(man); w.eval()
    traced = torch.jit.trace(w, dummy)
    traced.save('/home/ubuntu/P3O-CBF/rl_sar/policy/g1/p3o_end2end/policy.pt')
    print('Saved policy.pt')

    m2 = torch.jit.load('/home/ubuntu/P3O-CBF/rl_sar/policy/g1/p3o_end2end/policy.pt', map_location='cpu')
    with torch.no_grad(): m2o = m2(dummy).numpy()
    print(f'Load: {m2o[0,:5]}')
    print(f'Save/Load diff: {abs(ref-m2o).max():.10f}')

    d2 = torch.randn(1,2400)
    with torch.no_grad(): r1=orig.act_inference(d2).numpy(); r2=m2(d2).numpy()
    print(f'Random diff: {abs(r1-r2).max():.10f}')

if __name__=='__main__': main()
