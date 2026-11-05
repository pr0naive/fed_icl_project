"""Offline tests: data carving, prompts, and every arm end to end with a fake LLM."""

from p2.config import Config
from p2.data_mmlu import prepare
from p2.prompts import build_messages
from p2.protocol import Runner
from p2.retrieval import Retriever
from tests_p2.fakes import FakeBackend, fake_items


def _setup(**kw):
    cfg = Config()
    cfg.embedder = "fake"
    cfg.num_rounds = 2
    for k, v in kw.items():
        setattr(cfg, k, v)
    items = fake_items()
    q, clients, ev, info = prepare(cfg, items)
    return cfg, items, q, clients, ev, info


def test_data_carving_is_disjoint_and_paper_sized():
    cfg, items, q, clients, ev, info = _setup(eval_size=100)
    assert len(q) == 114                       # 2 per subject x 57
    assert len({x.subject for x in q}) == 57
    ids = lambda xs: {x.question for x in xs}
    pool = set().union(*[ids(c) for c in clients])
    assert not (ids(q) & pool) and not (ids(ev) & pool) and not (ids(q) & ids(ev))
    assert len(pool) + len(q) + len(ev) == len(items)


def test_queries_fixed_across_seeds_but_partition_changes():
    _, _, q1, c1, _, _ = _setup(seed=1)
    _, _, q2, c2, _, _ = _setup(seed=2)
    assert [x.question for x in q1] == [x.question for x in q2]
    assert [len(c) for c in c1] != [len(c) for c in c2]


def test_prompt_never_reveals_query_answer():
    cfg, items, q, clients, *_ = _setup()
    msgs = build_messages(clients[0][:3], q[0])
    assert msgs[0]["role"] == "system"
    assert [m["role"] for m in msgs[1:]] == ["user", "assistant"] * 3 + ["user"]
    assert msgs[-1]["content"].endswith("Answer:")


def test_all_arms_run_and_log():
    cfg, items, q, clients, ev, _ = _setup(eval_size=60)
    r = Runner(cfg, FakeBackend(items), Retriever("fake"))
    for arm in ["zeroshot", "gt", "ub", "randlabel", "fedicl", "fedicl_free", "solo_free"]:
        r.trace = []
        out = r.run_arm(arm, q, clients, ev)
        assert 0 <= out["accuracy"] <= 1 and len(out["per_query_correct"]) == 114
        assert r.trace, arm
        if arm in ("fedicl", "fedicl_free", "solo_free"):
            assert len(out["rounds"]) == cfg.num_rounds + 1
            last = out["rounds"][-1]
            n_clients = 1 if arm == "solo_free" else 3
            assert len(last["client_preds"]) == n_clients
            assert len(last["pseudo_label_acc"]) == n_clients
            assert out["heldout_accuracy"] is not None


def test_both_pool_gives_2C_demos_and_union_gives_C():
    for pool, expected in [("both", 10), ("union", 5)]:
        cfg, items, q, clients, *_ = _setup(prompt_pool=pool, num_rounds=1)
        r = Runner(cfg, FakeBackend(items), Retriever("fake"))
        r.loop(q, clients, "full", "fedicl")
        query_calls = [t for t in r.trace if t["phase"] == "query"]
        assert max(t["n_demos"] for t in query_calls) == expected


def test_per_seed_queries_differ_and_fixed_mode_does_not():
    _, _, q1, *_ = _setup(seed=1, query_mode="per_seed")
    _, _, q2, *_ = _setup(seed=2, query_mode="per_seed")
    assert [x.question for x in q1] != [x.question for x in q2] and len(q1) == len(q2) == 114
