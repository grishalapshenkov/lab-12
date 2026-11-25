"""
Лабораторная работа № 12. Вариант 14.
Составление расписания на 3 машинах: минимизация времени завершения всех работ.
Точный метод — полный перебор; конструктивная эвристика — «наименее загруженная
машина»; локальный поиск — перенос работы. Размерности: 9…15 (точно), 200 (эвристики).
Дополнительное задание М7: прикладная постановка и экономический эффект.
"""
import gc
import json
import math
import random
import statistics
import time

M = 3                      # число машин
LO, HI = 1, 50             # длительности работ — случайные целые из [1; 50]


# ------------------------------------------------------------ вспомогательное
def random_jobs(n, rng):
    return [rng.randint(LO, HI) for _ in range(n)]


def loads_of(jobs, assign, m=M):
    loads = [0] * m
    for p, k in zip(jobs, assign):
        loads[k] += p
    return loads


def lower_bound(jobs, m=M):
    """Нижняя граница оптимума: max(самая длинная работа, ⌈сумма / m⌉)."""
    return max(max(jobs), math.ceil(sum(jobs) / m))


# ------------------------------------------------------------ точные методы
def brute_force(jobs, m=M):
    """Полный перебор: все назначения работ на m машин. Первая работа фиксируется на
    машине 0 (машины одинаковы) — рассматривается m^(n−1) назначений.
    Возвращает (оптимальное время завершения, число рассмотренных назначений)."""
    n = len(jobs)
    best = [sum(jobs)]
    count = [0]
    loads = [0] * m

    def rec(i):
        if i == n:
            count[0] += 1
            mx = max(loads)
            if mx < best[0]:
                best[0] = mx
            return
        p = jobs[i]
        for k in range(m):
            loads[k] += p
            rec(i + 1)
            loads[k] -= p

    loads[0] = jobs[0]
    rec(1)
    return best[0], count[0]


def dp_optimum(jobs, upper):
    """Независимое точное решение (псевдополиномиальное ДП по множеству состояний —
    упорядоченных троек загрузок). Состояния с загрузкой выше верхней оценки отбрасываются.
    Используется для проверки перебора и для размерностей, где перебор слишком долог."""
    states = {(0, 0, 0)}
    for p in jobs:
        new = set()
        for a, b, c in states:
            for t in ((a + p, b, c), (a, b + p, c), (a, b, c + p)):
                if t[0] <= upper and t[1] <= upper and t[2] <= upper:
                    new.add(tuple(sorted(t, reverse=True)))
        states = new
    return min(s[0] for s in states)


# ---------------------------------------------------------------- эвристики
def least_loaded(jobs, m=M):
    """Конструктивная эвристика: каждая очередная работа (в заданном порядке)
    назначается на наименее загруженную машину. Θ(n·m)."""
    loads = [0] * m
    assign = []
    for p in jobs:
        k = loads.index(min(loads))
        assign.append(k)
        loads[k] += p
    return assign


def lpt(jobs, m=M):
    """Для сравнения: та же эвристика, но работы предварительно упорядочены по убыванию."""
    order = sorted(range(len(jobs)), key=lambda i: -jobs[i])
    loads = [0] * m
    assign = [0] * len(jobs)
    for i in order:
        k = loads.index(min(loads))
        assign[i] = k
        loads[k] += jobs[i]
    return assign


def local_search_move(jobs, assign, m=M):
    """Локальный поиск с окрестностью «перенос одной работы на другую машину».
    Перенос принимается, если он уменьшает пару (время завершения, сумма квадратов
    загрузок) в лексикографическом порядке — это гарантирует завершение.
    Возвращает (новое назначение, число проходов, число принятых переносов)."""
    assign = list(assign)
    loads = loads_of(jobs, assign, m)
    improved, passes, moves = True, 0, 0
    while improved:
        improved = False
        passes += 1
        for i, p in enumerate(jobs):
            src = assign[i]
            for dst in range(m):
                if dst == src:
                    continue
                cur_max = max(loads)
                new_src, new_dst = loads[src] - p, loads[dst] + p
                rest = max((loads[k] for k in range(m) if k != src and k != dst), default=0)
                new_max = max(new_src, new_dst, rest)
                old_sq = loads[src] ** 2 + loads[dst] ** 2
                new_sq = new_src ** 2 + new_dst ** 2
                if new_max < cur_max or (new_max == cur_max and new_sq < old_sq):
                    loads[src], loads[dst] = new_src, new_dst
                    assign[i] = dst
                    improved = True
                    moves += 1
                    break
    return assign, passes, moves


def random_assignment(jobs, rng, m=M):
    return [rng.randrange(m) for _ in jobs]


def makespan(jobs, assign, m=M):
    return max(loads_of(jobs, assign, m))


# ---------------------------------------------------------------- замеры
def timed(func, *args, repeats=3):
    times = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times), result


def make_plots(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ex = res["exact"]
    ns = [r["n"] for r in ex]
    plt.figure(figsize=(6.8, 4.4))
    plt.semilogy(ns, [r["t"] for r in ex], "o-", label="время полного перебора")
    base = ex[0]["t"] / 3 ** (ex[0]["n"] - 1)
    plt.semilogy(ns, [base * 3 ** (n - 1) for n in ns], "k--", lw=1, label=r"$c \cdot 3^{n-1}$")
    plt.axhline(10, color="gray", ls=":", label="10 с")
    plt.xlabel("число работ n"); plt.ylabel("время, с (логарифмическая шкала)")
    plt.title("Рост времени точного метода"); plt.grid(True, which="both", alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig("img/lr12_exact.png", dpi=170); plt.close()

    dv = res["series"]
    plt.figure(figsize=(6.8, 4.2))
    ns2 = [r["n"] for r in dv]
    x = range(len(ns2))
    plt.bar([i - 0.2 for i in x], [r["dev_ll"] for r in dv], 0.4, label="наименее загруженная машина")
    plt.bar([i + 0.2 for i in x], [r["dev_ls"] for r in dv], 0.4, label="+ перенос работы")
    plt.xticks(list(x), ns2); plt.xlabel("число работ n"); plt.ylabel("среднее отклонение от оптимума, %")
    plt.title("Качество эвристик (30 наборов на каждое n)"); plt.legend(); plt.grid(True, axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig("img/lr12_dev.png", dpi=170); plt.close()

    ap = res["app"]
    plt.figure(figsize=(6.8, 4.2))
    plt.boxplot([ap["days"]["ll"], ap["days"]["ls"], ap["days"]["lpt"]],
                tick_labels=["наименее\nзагруженная", "+ перенос\nработы", "сортировка\nпо убыванию"], showmeans=True)
    plt.ylabel("время завершения смены, мин"); plt.title("250 рабочих дней, 200 заказов в день")
    plt.grid(True, axis="y", alpha=.3); plt.tight_layout(); plt.savefig("img/lr12_app.png", dpi=170); plt.close()


# ------------------------------------------------------------------- main
if __name__ == "__main__":
    res = {}
    rng = random.Random(14)

    # --- 1. Рост времени полного перебора
    res["exact"] = []
    print(f"{'n':>3}{'назначений':>14}{'время, с':>11}")
    n = 8
    while True:
        n += 1
        jobs = random_jobs(n, rng)
        t, (opt, cnt) = timed(brute_force, jobs, repeats=1)
        assert cnt == 3 ** (n - 1)
        ll = makespan(jobs, least_loaded(jobs))
        assert opt == dp_optimum(jobs, ll), "перебор и ДП расходятся"
        res["exact"].append({"n": n, "count": cnt, "t": t, "opt": opt})
        print(f"{n:>3}{cnt:>14}{t:>11.3f}", flush=True)
        if t > 10:
            res["limit"] = n
            break

    # --- 2. Отклонение эвристик от оптимума, 30 наборов на каждое n = 9…15
    res["series"] = []
    print(f"\n{'n':>3}{'LL, %':>9}{'LL+move, %':>12}{'доля опт. LL':>14}{'доля опт. LL+move':>19}{'источник оптимума':>20}")
    for n in range(9, 16):
        d_ll, d_ls, hit_ll, hit_ls = [], [], 0, 0
        source = "перебор" if n <= 13 else "ДП"
        for _ in range(30):
            jobs = random_jobs(n, rng)
            a1 = least_loaded(jobs)
            c1 = makespan(jobs, a1)
            a2, _, _ = local_search_move(jobs, a1)
            c2 = makespan(jobs, a2)
            opt = brute_force(jobs)[0] if n <= 13 else dp_optimum(jobs, c1)
            if n <= 13:
                assert opt == dp_optimum(jobs, c1)          # перекрёстная проверка
            assert opt <= c2 <= c1
            d_ll.append((c1 - opt) / opt * 100)
            d_ls.append((c2 - opt) / opt * 100)
            hit_ll += c1 == opt
            hit_ls += c2 == opt
        res["series"].append({"n": n, "dev_ll": statistics.mean(d_ll), "dev_ls": statistics.mean(d_ls),
                              "max_ll": max(d_ll), "max_ls": max(d_ls),
                              "hit_ll": hit_ll / 30, "hit_ls": hit_ls / 30, "source": source})
        r = res["series"][-1]
        print(f"{n:>3}{r['dev_ll']:>9.2f}{r['dev_ls']:>12.2f}{r['hit_ll'] * 100:>13.0f}%{r['hit_ls'] * 100:>18.0f}%{source:>20}", flush=True)

    # --- 3. Большая размерность n = 200
    N = 200
    jobs = random_jobs(N, random.Random(200))
    lb = lower_bound(jobs)
    res["large"] = {"n": N, "sum": sum(jobs), "lb": lb}
    rnd = random_assignment(jobs, random.Random(1))
    t1, a1 = timed(least_loaded, jobs, repeats=20)
    t2, (a2, passes, moves) = timed(local_search_move, jobs, a1, repeats=20)
    res["large"]["rows"] = {
        "random": {"c": makespan(jobs, rnd), "t": None},
        "ll": {"c": makespan(jobs, a1), "t": t1},
        "ls": {"c": makespan(jobs, a2), "t": t2, "passes": passes, "moves": moves}}
    print(res["large"])
    # усреднение по 30 наборам
    agg = {"random": [], "ll": [], "ls": []}
    r2 = random.Random(77)
    for _ in range(30):
        jb = random_jobs(N, r2)
        b = lower_bound(jb)
        x1 = least_loaded(jb)
        x2, _, _ = local_search_move(jb, x1)
        agg["random"].append((makespan(jb, random_assignment(jb, r2)) - b) / b * 100)
        agg["ll"].append((makespan(jb, x1) - b) / b * 100)
        agg["ls"].append((makespan(jb, x2) - b) / b * 100)
    res["large"]["avg_gap"] = {k: statistics.mean(v) for k, v in agg.items()}
    print(res["large"]["avg_gap"])

    # --- 4. М7: заказы на трёх упаковочных линиях, 250 рабочих дней
    r3 = random.Random(250)
    days = {"ll": [], "ls": [], "lpt": [], "lb": []}
    for _ in range(250):
        jb = random_jobs(200, r3)
        x1 = least_loaded(jb)
        x2, _, _ = local_search_move(jb, x1)
        days["ll"].append(makespan(jb, x1))
        days["ls"].append(makespan(jb, x2))
        days["lpt"].append(makespan(jb, lpt(jb)))
        days["lb"].append(lower_bound(jb))
    res["app"] = {"days": days, "mean": {k: statistics.mean(v) for k, v in days.items()}}
    print(res["app"]["mean"])

    with open("res/lr12.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    make_plots(res)
