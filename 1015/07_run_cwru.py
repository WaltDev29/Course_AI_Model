"""
================================================================================
 [08장 보충] 시계열 이상탐지 — CWRU 베어링 진동 데이터
================================================================================
 과목 : 인공지능 모델실습 · 08장 시계열 데이터와 순환 신경망 보충 · 황종철

 무엇을 하는가
   모터 베어링의 진동(시계열)을 보고 "평소와 다른가"를 판정한다.
   결함 데이터는 학습에 한 창도 쓰지 않는다. 정상 진동만 저장해 두고,
   새 신호가 저장된 정상과 얼마나 먼지(거리)로 이상 여부를 정한다.

 처리 흐름 (5단계)
   ① 진동 신호   48 kHz → 12 kHz 로 줄임
   ② 창 나누기   2048점(0.17초)씩 자르고 절반(1024점)씩 겹침
   ③ 특징 추출   창 하나 → 계산식 특징 19개 (log RMS · log 첨도 · log 파고율 · 대역 에너지 비율 16개)
   ④ 이상 점수   표준화한 특징과 "정상 특징 은행" 사이의 최근접 거리 (창마다 점수 하나)
   ⑤ 판정       점수가 기준값보다 크면 이상
                기준값 A = 정상 은행 안 최근접 거리(자기 자신 제외)의 99백분위 × 1.2  (정상만으로 정함)
                기준값 B = 20  (학습에 없던 부하 2·3 HP 의 정상이 A 를 넘는 것을 보고 올린 값)

 데이터 (CWRU Bearing Data Center 공개 데이터, .mat)
   기본 경로 : 이 파일 기준 ../../../data/cwru  (강의자료\\data\\cwru)
   다른 곳에 두었으면 --root 로 지정한다. 필요한 파일 13개 :
     Normal/97_Normal_0.mat … 100_Normal_3.mat   정상 (부하 0·1·2·3 HP, 48 kHz)
     48k_Drive_End_Bearing_Fault_Data/IR·B·OR … (부하 0 HP, 결함 지름 0.18 · 0.36 · 0.53 mm)

 사용법 (아나콘다 프롬프트에서)
   python 8_10_run_cwru.py            # train → test → envelope → slide 순서로 모두 실행 (수 초)
   python 8_10_run_cwru.py train      # 정상(0·1 HP) 앞 70% 창으로 특징 은행 · 표준화 값 · 기준값 A 저장
   python 8_10_run_cwru.py test       # 정상 뒤 30% · 미학습 부하 정상 · 결함 9종을 채점 → 시간축 · 분포 그림
   python 8_10_run_cwru.py envelope   # 포락선 스펙트럼 : 검출된 이상이 물리적으로 맞는지 확인 (BPFO · BPFI)
   python 8_10_run_cwru.py slide      # 파형 · 포락선 · 점수 분포를 한 장에 모은 요약 그림

 결과 (이 파일 옆 out_cwru 폴더)
   bank.npz · calib.json          정상 특징 은행 · 평균/편차 · 기준값
   scores.csv                     창 단위 점수표 (class, window, score)
   vib_timeline.png               정상 · 결함 신호의 창별 점수 시간축
   score_dist.png                 클래스별 점수 분포 + 기준값 A/B
   envelope.png                   외륜 · 내륜 결함의 포락선 스펙트럼
   fig21_cwru.png                 요약 그림 (파형 · 충격 간격 · 포락선 · 분포)
   run_log.txt                    화면 출력 사본 (실행할 때마다 뒤에 이어 붙음)

 필요 패키지 : numpy scipy matplotlib   (텐서플로 불필요)

 모든 강의 자료는 저작권 있음으로 외부 공유 절대 금지
================================================================================
"""
import argparse, os, sys, json, csv, time
import numpy as np
import scipy.io as sio            # .mat (매트랩) 파일 읽기
from scipy import signal          # 다운샘플링 · 필터 · 힐베르트 변환
import matplotlib
matplotlib.use("Agg")             # 창을 띄우지 않고 그림 파일로만 저장
import matplotlib.pyplot as plt
import matplotlib.font_manager

# 데이터 위치 : SRC\source_1\ch8 에서 세 단계 위로 올라가면 강의자료 폴더
DATA_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "cwru"))
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_cwru")

# 그래프 한글 글꼴 : 윈도는 맑은 고딕, 리눅스는 Noto/나눔을 찾아 씀
for _p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
    if os.path.exists(_p): matplotlib.font_manager.fontManager.addfont(_p)          # 리눅스 대비
for _name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(f.name == _name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _name
        break
plt.rcParams["axes.unicode_minus"] = False
NAVY = "#15213F"; BLUE = "#3257B8"; GRAY = "#52586B"; RED = "#D93025"; GREEN = "#1F8A4C"; AMBER = "#B7791F"

# ── 주요 설정값 ────────────────────────────────────────────────────────────────
FS = 12000            # 분석 표본화율 (48 kHz 파일은 4:1 로 줄인다)
W = 2048; HOP = 1024  # 창 길이 2048점 = 0.17초 · 1024점씩 이동(절반 겹침)
THR_B = 20.0          # 운영 조건을 반영해 올린 기준값 (선택값)

# 정상 데이터 : 부하(HP) → 파일 이름.  부하가 클수록 회전 속도가 조금 느려짐 (1797 → 1730 rpm)
NORMALS = {0: "97_Normal_0", 1: "98_Normal_1", 2: "99_Normal_2", 3: "100_Normal_3"}

# 결함 데이터 9종 : 부하 0 HP · 48 kHz · 구동측(Drive End) 베어링
#   내륜(IR) · 볼(B) · 외륜(OR) × 결함 지름 0.007 · 0.014 · 0.021 인치 (= 0.18 · 0.36 · 0.53 mm)
#   외륜 결함은 6시 방향(@6) 파일을 사용
FAULTS = {
    "내륜 0.18 mm": "48k_Drive_End_Bearing_Fault_Data/IR/007/109_0.mat", "내륜 0.36 mm": "48k_Drive_End_Bearing_Fault_Data/IR/014/174_0.mat", "내륜 0.53 mm": "48k_Drive_End_Bearing_Fault_Data/IR/021/213_0.mat",
    "볼 0.18 mm": "48k_Drive_End_Bearing_Fault_Data/B/007/122_0.mat", "볼 0.36 mm": "48k_Drive_End_Bearing_Fault_Data/B/014/189_0.mat", "볼 0.53 mm": "48k_Drive_End_Bearing_Fault_Data/B/021/226_0.mat",
    "외륜 0.18 mm": "48k_Drive_End_Bearing_Fault_Data/OR/007/@6/135_0.mat", "외륜 0.36 mm": "48k_Drive_End_Bearing_Fault_Data/OR/014/201@6_0.mat", "외륜 0.53 mm": "48k_Drive_End_Bearing_Fault_Data/OR/021/@6/238_0.mat"}
# ※ 174_0.mat(내륜 0.36 mm)은 원본 파일 자체가 짧아(약 1.3초) 창이 14개만 나온다.


class Tee:
    """print 출력을 화면과 run_log.txt 에 동시에 기록"""
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8"); self.o = sys.stdout
    def write(self, s): self.o.write(s); self.f.write(s)
    def flush(self): self.o.flush(); self.f.flush()


# ================================================================ ①② 신호 → 창 → ③ 특징
def load_de(path):
    """.mat 에서 구동측(DE) 가속도를 읽고 48 kHz 이면 12 kHz 로 줄인다"""
    if not os.path.exists(path): raise SystemExit(f"[오류] 파일이 없습니다: {path}\n       DATA_ROOT 또는 --root 를 확인하세요.")
    d = sio.loadmat(path)
    # 변수 이름은 파일마다 다르다 (예: X097_DE_time). 'DE_time' 으로 끝나는 것을 찾는다.
    # 99_Normal_2.mat 에는 X098·X099 두 개가 들어 있어 마지막 것(X099)을 쓴다 → [-1]
    k = [x for x in d if x.endswith("DE_time")][-1]
    x = d[k].ravel().astype(np.float64)                       # (N, 1) 배열 → 1차원, 단위는 g (중력가속도)
    # 정상·48k 파일은 48 kHz 이므로 4:1 로 줄여 모든 신호를 12 kHz 로 맞춤
    # decimate 는 줄이기 전에 저역통과 필터를 걸어 엘리어싱(고주파가 저주파로 섞이는 현상)을 막는다
    if "48k" in path or "Normal" in path: x = signal.decimate(x, 4, zero_phase=True)
    return x


def windows(x):
    """긴 신호를 2048점짜리 창으로 자른다 (1024점씩 이동 → 절반 겹침). 결과: (창 수, 2048)"""
    return np.stack([x[i:i + W] for i in range(0, len(x) - W + 1, HOP)])


def feats(win):
    """창 하나 → 특징 19개. 학습이 아니라 계산식이다

       RMS      : 진동의 평균 세기          (결함이 생기면 커짐)
       첨도     : 분포의 뾰족함, 정규분포=3   (짧은 충격이 섞이면 커짐)
       파고율   : 최댓값 / RMS               (튀는 값이 있으면 커짐)
       대역 에너지 비율 16개 : 0~6 kHz 를 375 Hz 씩 16칸으로 나눠 각 칸의 에너지 비율
                  (결함 충격은 특정 고주파 대역의 에너지를 키움)
       값의 범위가 수십 배씩 차이 나므로 모두 log10 을 취한다."""
    rms = np.sqrt((win ** 2).mean(1)); pk = np.abs(win).max(1); crest = pk / rms
    m = win - win.mean(1, keepdims=True); kurt = (m ** 4).mean(1) / ((m ** 2).mean(1) ** 2)
    # 해닝 창을 곱한 뒤 FFT → 주파수별 에너지 (창 끝이 잘려 생기는 가짜 주파수를 줄이기 위함)
    spec = np.abs(np.fft.rfft(win * np.hanning(W), axis=1)) ** 2; freqs = np.fft.rfftfreq(W, 1 / FS)
    edges = np.linspace(0, FS / 2, 17)                        # 0 ~ 6000 Hz 를 16칸으로
    bands = np.stack([spec[:, (freqs >= a) & (freqs < b)].sum(1) for a, b in zip(edges[:-1], edges[1:])], 1)
    bands = np.log10(bands / bands.sum(1, keepdims=True) + 1e-12)   # 전체 대비 비율 → 세기와 무관한 '모양'
    return np.column_stack([np.log10(rms), np.log10(kurt), np.log10(crest), bands])


# ================================================================ ④ 이상 점수
def nn_dist(Z, B):
    """표준화된 특징 Z(창 수 × 19) 와 은행 B(121 × 19) 사이의 최근접 거리
       Z[:, None, :] - B[None, :, :] → (창 수, 121, 19) 로 모든 쌍의 차이를 한 번에 계산(브로드캐스팅)
       → 유클리드 거리 (창 수, 121) → 행마다 최솟값 = 가장 비슷한 정상 창까지의 거리"""
    return np.sqrt(((Z[:, None, :] - B[None, :, :]) ** 2).sum(2)).min(1)


def load_model(out):
    """train 에서 저장한 특징 은행 · 평균/편차 · 기준값을 불러온다"""
    p = os.path.join(out, "bank.npz"); c = os.path.join(out, "calib.json")
    if not (os.path.exists(p) and os.path.exists(c)): raise SystemExit("[오류] 학습 산출물이 없습니다. 먼저 'python 8_10_run_cwru.py train' 을 실행하세요.")
    z = np.load(p); cal = json.load(open(c, encoding="utf-8"))
    return z["B"], np.array(cal["mu"]), np.array(cal["sd"]), cal


# ================================================================ train : 정상만으로 은행 · 기준값 A 만들기
def cmd_train(root, out):
    t0 = time.time()
    wn = {hp: windows(load_de(os.path.join(root, "Normal", f"{n}.mat"))) for hp, n in NORMALS.items()}
    # 학습 : 0·1 HP 정상의 앞 70% 만 사용. 뒤 30% 는 test 에서 '처음 보는 정상'으로 쓴다
    # (시계열이므로 무작위로 섞지 않고 시간 순서대로 앞·뒤를 나눈다)
    tr = np.concatenate([wn[0][:int(len(wn[0]) * .7)], wn[1][:int(len(wn[1]) * .7)]])
    print(f"[학습] 정상 진동만 사용 — 부하 0·1 HP 앞 70% = {len(tr)}창 (창 하나 = {W}점 = {W / FS:.2f}초) · 결함 신호 0창")
    # 표준화 : 특징마다 눈금이 다르므로 (특징 − 평균) / 편차 로 맞춘다. 평균·편차는 학습 정상에서만 구함
    F_tr = feats(tr); mu, sd = F_tr.mean(0), F_tr.std(0) + 1e-6
    B = (F_tr - mu) / sd                                      # 정상 특징 은행 (121창 × 19)
    print(f"[완료] 특징 {F_tr.shape[1]}개 × {len(B)}창을 표준화해 저장 — {time.time() - t0:.1f}초 (가중치 학습 없음 : 특징을 '저장'만 했습니다)")
    # 기준값 A : 은행 안에서 각 창의 '자기 자신을 뺀' 최근접 거리를 구하고 (대각선을 무한대로)
    #           그 99백분위에 여유 20%(× 1.2)를 둔다 → 정상끼리의 거리보다 조금 큰 값
    d = np.sqrt(((B[:, None, :] - B[None, :, :]) ** 2).sum(2)); np.fill_diagonal(d, np.inf); loo = d.min(1)
    thr = float(np.percentile(loo, 99) * 1.2)
    print(f"[기준값] A = 정상 은행 안 최근접 거리(자기 제외) 99백분위 {np.percentile(loo, 99):.2f} × 1.2 = {thr:.2f}  ← 정상만으로 정한 출발값")
    print(f"         B = {THR_B:.0f}  ← 학습에 없던 부하의 정상이 A 를 넘는 것을 보고 올린 값 (test 에서 확인)")
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "bank.npz"), B=B)
    json.dump({"mu": mu.tolist(), "sd": sd.tolist(), "thr_A": thr, "thr_B": THR_B, "n_bank": int(len(B))}, open(os.path.join(out, "calib.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ================================================================ test : ⑤ 정상 · 결함 채점과 판정
def cmd_test(root, out):
    B, mu, sd, cal = load_model(out); thrA, thrB = cal["thr_A"], cal["thr_B"]
    score = lambda win: nn_dist((feats(win) - mu) / sd, B)   # 창 묶음 → 창마다 이상 점수
    wn = {hp: windows(load_de(os.path.join(root, "Normal", f"{n}.mat"))) for hp, n in NORMALS.items()}
    held = np.concatenate([wn[0][int(len(wn[0]) * .7):], wn[1][int(len(wn[1]) * .7):]])   # 검증 : 학습에 안 쓴 뒤 30%
    # 채점 대상 12묶음 : 처음 보는 정상(같은 부하) · 학습에 없던 부하 2·3 HP 정상 · 결함 9종
    groups = [("정상 · 학습 부하 (0·1 HP)", score(held), GREEN), ("정상 · 미학습 부하 2 HP", score(wn[2]), AMBER), ("정상 · 미학습 부하 3 HP", score(wn[3]), AMBER)]
    fault_scores = {}
    for name, rel in FAULTS.items():
        fault_scores[name] = score(windows(load_de(os.path.join(root, rel)))); groups.append((name, fault_scores[name], RED))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "scores.csv"), "w", newline="", encoding="utf-8-sig") as fp:   # 엑셀에서 한글이 깨지지 않도록 utf-8-sig
        w = csv.writer(fp); w.writerow(["class", "window", "score"])
        for name, v, _ in groups:
            for i, s in enumerate(v): w.writerow([name, i, round(float(s), 3)])
    print(f"[채점] 이상 점수 = 정상 특징 은행({cal['n_bank']}창)과의 최근접 거리 · 기준값 A {thrA:.2f} / B {thrB:.0f}")
    print(f"  {'구분':22s} {'창수':>5s} {'최소':>7s} {'중앙':>7s} {'최대':>7s}   A초과   B초과")
    for name, v, _ in groups:
        print(f"  {name:22s} {len(v):5d} {v.min():7.2f} {np.median(v):7.2f} {v.max():7.2f}  {int((v > thrA).sum()):4d}/{len(v):<4d} {int((v > thrB).sum()):4d}/{len(v)}")
    allf = np.concatenate(list(fault_scores.values())); unseen = np.concatenate([groups[1][1], groups[2][1]]); heldS = groups[0][1]
    print()
    # 결함 창이 기준값을 넘으면 '검출'(재현율), 정상 창이 기준값을 넘으면 '오판'(오경보)
    print(f"[결과] 결함 9종 {len(allf)}창 : A 초과 {int((allf > thrA).sum())} · B 초과 {int((allf > thrB).sum())} (결함 최소 점수 {allf.min():.1f})")
    print(f"       학습 부하 정상 {len(heldS)}창 : A 오판 {int((heldS > thrA).sum())} · 미학습 부하(2·3 HP) 정상 {len(unseen)}창 : A 오판 {int((unseen > thrA).sum())}, B 오판 {int((unseen > thrB).sum())}")
    print(f"       → 정상만으로 정한 A 는 운영 조건(부하)이 바뀌면 정상을 오판한다. 기준값을 B 로 올리거나, 2·3 HP 정상을 학습에 넣는다 (보충 자료 참고).")

    # 그림 1 : 정상 · 결함 신호의 창별 점수 시간축 (앞 40창 = 약 3.4초)
    n_show = 40; sn = heldS[:n_show]; sf = fault_scores["내륜 0.18 mm"][:n_show]; tt = np.arange(n_show) * HOP / FS
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(tt, sn, "-o", color=GREEN, ms=3, lw=1, label="정상 신호의 창별 이상 점수"); ax.plot(tt, sf, "-o", color=RED, ms=3, lw=1, label="결함(내륜 0.18 mm) 신호의 창별 이상 점수")
    ax.axhline(thrA, color=BLUE, ls="--", lw=1.5); ax.text(tt[-1], thrA * 1.08, f"기준값 A {thrA:.2f}", color=BLUE, ha="right", fontsize=10, fontweight="bold")
    ax.set_yscale("log"); ax.set_xlabel("시간 (초, 0.17초 창)"); ax.set_ylabel("이상 점수 (최근접 거리)"); ax.legend(frameon=False, fontsize=9)
    ax.set_title("정상 진동 특징을 저장해 두고, 새 신호가 얼마나 다른지 거리로 판정 (창마다 점수 하나)", fontsize=11, color=NAVY)
    plt.tight_layout(); plt.savefig(os.path.join(out, "vib_timeline.png"), dpi=150); plt.close()
    # 그림 2 : 클래스별 분포 + A/B (점 하나 = 창 하나, 가로축 로그 눈금)
    fig, ax = plt.subplots(figsize=(9, 4.4)); rng = np.random.default_rng(0)
    for i, (lab, v, c) in enumerate(groups):
        y = len(groups) - 1 - i; ax.scatter(v, y + rng.uniform(-0.18, 0.18, len(v)), s=14, color=c, alpha=0.55)   # 세로로 살짝 흩어 겹침 방지
    ax.set_yticks(range(len(groups))); ax.set_yticklabels([g[0] for g in groups][::-1], fontsize=9)
    ax.axvline(thrA, color=BLUE, lw=2, ls="--"); ax.text(thrA * 1.05, len(groups) - 0.6, f"기준값 A {thrA:.1f}", color=BLUE, fontsize=10, fontweight="bold")
    ax.axvline(thrB, color=NAVY, lw=2, ls="-."); ax.text(thrB * 1.05, len(groups) - 0.6, f"기준값 B {thrB:.0f}", color=NAVY, fontsize=10, fontweight="bold")
    ax.set_xscale("log"); ax.set_xlabel("이상 점수 (창 단위 최근접 거리)"); ax.set_title("클래스별 점수 분포 : 결함은 전부 오른쪽, 미학습 부하의 정상만 A 와 B 사이", fontsize=11, color=NAVY)
    plt.tight_layout(); plt.savefig(os.path.join(out, "score_dist.png"), dpi=150); plt.close()
    print("saved", os.path.join(out, "scores.csv"), "/ vib_timeline.png / score_dist.png")


# ================================================================ envelope : 검출 결과의 물리적 확인
# 베어링 결함 특성 주파수 = 축 회전 주파수 × 베어링 형상으로 정해지는 계수
#   CWRU 구동측 베어링 SKF 6205-2RS JEM : 외륜 결함 BPFO = 3.5848 × fr, 내륜 결함 BPFI = 5.4152 × fr
#   1797 rpm → fr = 29.95 Hz → BPFO ≈ 107.4 Hz, BPFI ≈ 162.2 Hz
RPM = 1797; FR = RPM / 60; BPFO = 3.5848 * FR; BPFI = 5.4152 * FR             # SKF 6205 결함 특성 주파수 계수


def env_spec(x, lo=2000, hi=5500, n=FS * 2):
    """대역통과(공진대) → 힐베르트 포락선 → FFT : 충격의 반복 주파수를 드러낸다
       결함 충격은 베어링·하우징의 고주파 공진(2~5.5 kHz)을 울린다 → 그 대역만 걸러
       '울림의 겉모양(포락선)'을 구하면, 충격이 몇 Hz 마다 반복되는지가 스펙트럼 봉우리로 보인다"""
    b, a = signal.butter(4, [lo / (FS / 2), hi / (FS / 2)], "band"); y = signal.filtfilt(b, a, x[:n])
    env = np.abs(signal.hilbert(y)); env -= env.mean()
    sp = np.abs(np.fft.rfft(env * np.hanning(len(env)))); f = np.fft.rfftfreq(len(env), 1 / FS); return f, sp / sp.max()


def cmd_envelope(root, out):
    """포락선 스펙트럼 : 결함 반복 주파수가 베어링 기하 계산값(BPFO · BPFI)과 맞는지 확인"""
    os.makedirs(out, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    print(f"[포락선] {RPM} rpm · SKF 6205 → BPFO {BPFO:.1f} Hz (외륜) · BPFI {BPFI:.1f} Hz (내륜)")
    for ax, (key, fc, name) in zip(axs, [("외륜 0.18 mm", BPFO, "BPFO"), ("내륜 0.18 mm", BPFI, "BPFI")]):
        f, sp = env_spec(load_de(os.path.join(root, FAULTS[key])))
        m = (f > fc * 0.9) & (f < fc * 1.1); peak = f[m][sp[m].argmax()]   # 계산값 ±10% 안에서 가장 큰 봉우리
        print(f"  {key} : 포락선 스펙트럼 최대 피크 {peak:.1f} Hz ↔ 계산값 {fc:.1f} Hz (차이 {abs(peak - fc):.1f} Hz)")
        ax.plot(f, sp, color=RED, lw=0.9); ax.set_xlim(0, 520)
        for h in range(1, 4): ax.axvline(fc * h, color=BLUE, ls=":", lw=1)     # 기본 주파수와 2·3배 배음
        ax.text(fc + 8, 0.9, f"{name} {fc:.0f} Hz ×1 ×2 ×3", color=BLUE, fontsize=9, fontweight="bold"); ax.set_title(f"{key} 포락선 스펙트럼", fontsize=10.5, color=RED); ax.set_xlabel("Hz")
    fig.suptitle("모델이 잡은 이상이 물리적으로 설명되는가 : 결함 특성 주파수와 배음이 일치", fontsize=11, color=NAVY)
    plt.tight_layout(); plt.savefig(os.path.join(out, "envelope.png"), dpi=150); plt.close()
    print("saved", os.path.join(out, "envelope.png"))


# ================================================================ slide : 요약 그림 한 장
def cmd_slide(root, out):
    """요약 그림 : 정상 · 결함 파형(±3 g, 충격 간격) · 포락선 스펙트럼 · 클래스별 점수 분포(A · B · 미학습 부하)"""
    B, mu, sd, cal = load_model(out); thrA, thrB = cal["thr_A"], cal["thr_B"]
    score = lambda win: nn_dist((feats(win) - mu) / sd, B)
    wn = {hp: windows(load_de(os.path.join(root, "Normal", f"{n}.mat"))) for hp, n in NORMALS.items()}
    held = np.concatenate([wn[0][int(len(wn[0]) * .7):], wn[1][int(len(wn[1]) * .7):]])
    x_ir = load_de(os.path.join(root, FAULTS["내륜 0.18 mm"]))
    groups = [("정상 · 학습 부하 0·1 HP", score(held), GREEN), ("정상 · 미학습 2 HP", score(wn[2]), AMBER), ("정상 · 미학습 3 HP", score(wn[3]), AMBER)]
    for name, rel in FAULTS.items(): groups.append((name, score(windows(load_de(os.path.join(root, rel)))), RED))
    x_n = wn[0][5]; x_f = x_ir[5 * HOP:5 * HOP + W]; t = np.arange(W) / FS * 1000   # 같은 위치(6번째 창)의 정상 · 결함 파형
    print(f"[그림] 정상 파형 최대 {np.abs(x_n).max():.3f} g · 결함 파형 최대 {np.abs(x_f).max():.3f} g ({np.abs(x_f).max() / np.abs(x_n).max():.1f}배) · 충격 간격 1/BPFI = {1000 / BPFI:.1f} ms")
    fig = plt.figure(figsize=(9.9, 4.85)); gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.25], hspace=0.7, wspace=0.25, left=0.15, right=0.98, top=0.9, bottom=0.12)
    ax = fig.add_subplot(gs[0, 0]); ax.plot(t, x_n, color=GREEN, lw=0.7); ax.set_ylim(-3.5, 3.5); ax.set_title(f"정상 진동 (1797 rpm · 최대 {np.abs(x_n).max():.2f} g)", fontsize=9.5, color=GREEN); ax.set_xlabel("ms", fontsize=8.5); ax.set_ylabel("g", fontsize=8.5); ax.tick_params(labelsize=8)
    ax = fig.add_subplot(gs[0, 1]); ax.plot(t, x_f, color=RED, lw=0.7); ax.set_ylim(-3.5, 3.5); ax.set_title(f"결함 진동 (내륜 0.18 mm · 최대 {np.abs(x_f).max():.2f} g)", fontsize=9.5, color=RED); ax.set_xlabel("ms", fontsize=8.5); ax.tick_params(labelsize=8)
    ax.set_xlim(0, 60); seg = x_f[: int(0.012 * FS)]; pk = np.argmax(np.abs(seg)) / FS * 1000       # 첫 충격 위치에서 시작
    for k in range(0, 9): ax.axvline(pk + k * 1000 / BPFI, color=GRAY, ls=":", lw=0.8)              # 1/BPFI 간격 점선
    ax.text(1, -3.3, f"점선 간격 {1000 / BPFI:.1f} ms = 1/BPFI (충격 반복 주기)", fontsize=8, color=GRAY, va="bottom")
    ax = fig.add_subplot(gs[0, 2]); f, sp = env_spec(x_ir); ax.plot(f, sp, color=RED, lw=0.8); ax.set_xlim(0, 520); ax.set_yticks([])
    for h in range(1, 4): ax.axvline(BPFI * h, color=BLUE, ls=":", lw=1)
    ax.text(BPFI + 8, 0.88, f"BPFI {BPFI:.0f} Hz ×1 ×2 ×3", color=BLUE, fontsize=8.5, fontweight="bold"); ax.set_title("포락선 스펙트럼 : 계산값과 일치 (물리적 설명)", fontsize=9.5, color=NAVY); ax.set_xlabel("Hz", fontsize=8.5); ax.tick_params(labelsize=8)
    ax = fig.add_subplot(gs[1, :]); rng = np.random.default_rng(0)
    for i, (lab, v, c) in enumerate(groups):
        y = len(groups) - 1 - i; ax.scatter(v, y + rng.uniform(-0.2, 0.2, len(v)), s=9, color=c, alpha=0.55)
    ax.set_yticks(range(len(groups))); ax.set_yticklabels([g[0] for g in groups][::-1], fontsize=8)
    ax.axvline(thrA, color=BLUE, lw=1.8, ls="--"); ax.text(thrA * 1.04, len(groups) - 0.55, f"기준값 A {thrA:.1f} (정상만으로 결정)", color=BLUE, fontsize=8.5, fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    ax.axvline(thrB, color=NAVY, lw=1.8, ls="-."); ax.text(thrB * 1.04, len(groups) - 0.55, f"기준값 B {thrB:.0f} (운영 조건 반영)", color=NAVY, fontsize=8.5, fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    ax.set_xscale("log"); ax.set_xticks([2, 3, 5, 10, 20, 50, 100]); ax.set_xticklabels(["2", "3", "5", "10", "20", "50", "100"], fontsize=8); ax.set_xlabel("이상 점수 (정상 특징 은행과의 최근접 거리, 창 단위)", fontsize=9, color=GRAY)
    ax.set_title("결함 9종 480창은 전부 A · B 오른쪽 (전부 검출) · 학습에 없던 부하의 정상만 A 와 B 사이 (A 에서 오판 → B 로 조정)", fontsize=9.5, color=NAVY, fontweight="bold", pad=8)
    for s_ in ["top", "right", "left"]: ax.spines[s_].set_visible(False)
    ax.grid(axis="x", color="#E5E8EE", lw=0.6); ax.set_axisbelow(True)
    plt.savefig(os.path.join(out, "fig21_cwru.png"), dpi=150); plt.close()
    print("saved", os.path.join(out, "fig21_cwru.png"))


def cmd_all(root, out):
    cmd_train(root, out); print(); cmd_test(root, out); print(); cmd_envelope(root, out); print(); cmd_slide(root, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="all", choices=["all", "train", "test", "envelope", "slide"])
    ap.add_argument("--root", default=DATA_ROOT); ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    sys.stdout = Tee(os.path.join(a.out, "run_log.txt"))
    print(f"===== 8_10_run_cwru.py {a.cmd}  ({time.strftime('%Y-%m-%d %H:%M:%S')}) =====")
    {"all": cmd_all, "train": cmd_train, "test": cmd_test, "envelope": cmd_envelope, "slide": cmd_slide}[a.cmd](a.root, a.out)
