import numpy as np, pandas as pd, engine as E, luck as L, json
from scipy import stats
mk=E.load_market(); pay=json.load(open("payload.json"))
print("WHY SHARPE RISES WHEN THE BEST DAYS COME OUT")
print("Sharpe = mean/SD. Dropping a huge positive outlier cuts the mean LINEARLY but")
print("cuts the SD QUADRATICALLY. On right-skewed P&L the SD falls faster.\n")
rows=[]
for k in E.SUBS:
    b=E.reprice(E.load_blotter(k),mk); w=tuple(pay["subs"][k]["window"])
    d=E.daily_pnl(b,"mid",w).values
    srt=np.sort(d)[::-1]; kept=srt[5:]
    m0,s0=d.mean(),d.std(ddof=1); m1,s1=kept.mean(),kept.std(ddof=1)
    rows.append(dict(sub=k, skew=round(float(stats.skew(d)),1),
        kurt=round(float(stats.kurtosis(d,fisher=False)),0),
        mean=round(m0,1), sd=round(s0,1), sh=round(L.sharpe(d),2),
        mean_drop=f"{(m1/m0-1)*100:+.1f}%", sd_drop=f"{(s1/s0-1)*100:+.1f}%",
        sh5=round(L.sharpe(kept),2),
        pnl_kept=f"{kept.sum()/d.sum()*100:.0f}%",
        d2z=f"{pay['subs'][k]['rungs']['mid']['robust']['d2z']}/{len(d)}"))
df=pd.DataFrame(rows); pd.set_option("display.width",240)
print(df.to_string(index=False))
print("\nRead the two right-hand columns, not the Sharpe: P&L retained and days-to-erase")
print("are monotone measures of concentration. Sharpe after truncation is not.")
