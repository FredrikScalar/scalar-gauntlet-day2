import numpy as np, pandas as pd, engine as E, trips as T, luck as L, json
from scipy import stats
mk=E.load_market(); bg=L.BookGrid(mk); pay=json.load(open("payload.json"))
print("WHY SHARPE MOVES THE WAY IT DOES WHEN THE BEST 5 DAYS COME OUT")
print("Sharpe = mean/SD. Removing days k=5 from n days:")
print("   mean multiplier = (1 - P&L share of top5) x n/(n-5)")
print("   SD   multiplier depends on the top5's share of TOTAL SQUARED DEVIATION")
print("Sharpe rises when the top5 carry MORE of the variance than of the mean.\n")
rows=[]
for k in E.SUBS:
    b=E.reprice(E.load_blotter(k),mk); t=T.round_trips(b,"mid")
    d=L.daily_from_trips(t,L.trip_pnl(t,bg,"mid"),tuple(pay["subs"][k]["window"]))
    n=len(d); srt=np.sort(d)[::-1]; top=srt[:5]; kept=srt[5:]
    m=d.mean(); sq=(d-m)**2
    pnl_share=top.sum()/d.sum()
    var_share=((top-m)**2).sum()/sq.sum()
    rows.append(dict(sub=k, n=n, sd=d.std(ddof=1), mean=m,
      pnl_share=float(pnl_share), var_share=float(var_share),
      mean_mult=float(kept.mean()/m), sd_mult=float(kept.std(ddof=1)/d.std(ddof=1)),
      sh=L.sharpe(d), sh5=L.sharpe(kept),
      skew=float(stats.skew(d)), kurt=float(stats.kurtosis(d,fisher=False)),
      cv=float(d.std(ddof=1)/m)))
df=pd.DataFrame(rows).sort_values("var_share",ascending=False)
pd.set_option("display.width",250)
print(f"{'sub':>13} {'n':>5} {'top5 % of P&L':>14} {'top5 % of VARIANCE':>19} "
      f"{'gap':>7} {'Sharpe':>7} {'-5 days':>8} {'moves':>7}")
for _,r in df.iterrows():
    gap=r.var_share-r.pnl_share
    print(f"{r['sub']:>13} {int(r['n']):>5} {r.pnl_share*100:>13.1f}% {r.var_share*100:>18.1f}% "
          f"{gap*100:>+6.1f} {r.sh:>7.2f} {r.sh5:>8.2f} {('UP' if r.sh5>r.sh else 'DOWN'):>7}")
print("\nThe 'gap' column is the whole story: variance share minus P&L share.")
print("Positive -> the top days were mostly noise, so removing them helps the ratio.")
print("Negative -> the top days were mostly the edge, so removing them hurts it.\n")
print(f"{'sub':>13} {'daily mean':>11} {'daily SD':>10} {'SD/mean':>8} {'skew':>6} {'kurt':>7}")
for _,r in df.iterrows():
    print("{:>13} {:>11.1f} {:>10.1f} {:>8.1f} {:>6.1f} {:>7.0f}".format(
        r["sub"], r["mean"], r["sd"], r["cv"], r["skew"], r["kurt"]))
json.dump(df.to_dict("records"),open("varshare.json","w"))
