import time, numpy as np
from pathlib import Path
from nettc.synth import SyntheticCorpus
from nettc.sessionize import sessionize
from nettc.store import Store
from nettc.dissect import STRATUM_NAMES
from nettc.encryption import CIPHER_NAMES, cipher_is_deprecated
import sys
out = Path(sys.argv[1])
corpus = SyntheticCorpus(n_classes=8, sessions_per_class=300, seed=1)
t=time.time(); caps = corpus.write(out)
print(f"wrote {len(caps)} pcaps in {time.time()-t:.1f}s; total {sum(p.stat().st_size for p,_ in caps)/1e6:.1f} MB")
names = corpus.class_names()
t=time.time()
stats = sessionize(caps, out/"store.h5", names, lambda p,g: names.index(g))
print(f"sessionize {time.time()-t:.1f}s")
for k,v in stats.as_dict().items():
    print(f"  {k}:", {STRATUM_NAMES[int(a)]: b for a,b in v.items()} if k=="stratum_counts" else v)
s = Store(out/"store.h5")
print(f"\nstore: n={s.n} classes={s.n_classes} ceiling={s.byte_ceiling}")
print("labels      :", np.bincount(s.label, minlength=8))
for f in ("prefix_len","true_length","packet_count"):
    a=getattr(s,f); print(f"{f:12}: min {a.min()} med {int(np.median(a))} max {a.max()}")
print("sessions below 1024B: %.3f   below 784B: %.3f" % ((s.prefix_len<1024).mean(), (s.prefix_len<784).mean()))
cs={}
for c in s.cipher_suite: cs[int(c)]=cs.get(int(c),0)+1
print("ciphers:", {CIPHER_NAMES.get(k, 'none' if k==0 else f'0x{k:04x}'): v for k,v in sorted(cs.items())})
print("deprecated-cipher share: %.3f" % float(np.mean([cipher_is_deprecated(int(c)) for c in s.cipher_suite])))
b = s.get_bytes(np.arange(min(400,s.n)))
print("MAC bytes zeroed :", bool((b[:,0:12]==0).all()))
print("IP addrs zeroed  :", bool((b[:,26:34]==0).all()))
print("ports retained   :", bool((b[:,34:38]!=0).any()))
print("ident nonzero    :", bool((s.get_ident(np.arange(5))!=0).any()))
print("pkt_offsets[0:3] :", s.pkt_offsets[:3,:6].tolist())
s.close()
