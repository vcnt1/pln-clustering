"""Trains the mood model (Approach A: TF-IDF + Ridge, ADR-0008) and registers it.

This is a deliberately lightweight MVP trainer: it builds a small synthetic,
in-memory labeled dataset instead of running the full offline pipeline
(ingest/labels/split with corpus files, hash-gated reports, and the
register/promote registry CLI from specs 02/03/05/07). It exists so mood-ml
has a real model to serve; the full spec-compliant pipeline remains
unimplemented (out of scope for this MVP).
"""

from __future__ import annotations

import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from transform.features import extract_features, features_to_row

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
FEATURE_SPEC_VERSION = "fs-1"
HISTORY_WINDOW = 30
HISTORY_SCOPE = "conversation"
ALGORITHM = "tfidf-ridge"

# Rubric levels per mood-ml/specs/01-dataset-contract.md §4.
# Deliberately varied phrasing (not just topic swaps) so TF-IDF picks up a
# broader vocabulary and generalizes past these exact templates.
_TOPICS = ["wifi", "ar-condicionado", "limpeza do quarto", "check-in", "a reserva", "o pagamento", "chuveiro", "estacionamento"]

_TEMPLATES: dict[float, list[str]] = {
    -1.0: [
        "É um ABSURDO, o problema com {topic} não foi resolvido, vou cancelar e reclamar publicamente!!!",
        "Isso é INACEITÁVEL, {topic} continua um caos e ninguém me responde. Quero meu dinheiro de volta!!!",
        "PÉSSIMO atendimento, {topic} nunca funcionou e vocês simplesmente ignoram meus contatos!!!",
        "Já chega! {topic} é um descaso total, vou processar vocês!!!",
        "Uma vergonha o que fizeram com {topic}. NUNCA MAIS fico aqui e vou avisar todo mundo!!!",
        "Vocês são incompetentes, {topic} está um lixo e o suporte não presta nenhuma atenção!!!",
        "Que descaso, {topic} nunca foi resolvido e o suporte simplesmente sumiu!!!",
        "Isso é revoltante, fui enganado sobre {topic} e quero reembolso AGORA!!!",
        "Nunca vi tanta incompetência, {topic} é um caos completo, uma vergonha!!!",
        "Estou furioso com {topic}, isso é o fim, vou denunciar vocês em todos os lugares!!!",
        "Detestei tudo, {topic} arruinou minha viagem e ninguém se importa!!!",
        "Que horror, {topic} é a pior experiência que já tive na vida!!!",
    ],
    -0.5: [
        "Já é a segunda vez que peço e {topic} ainda não foi resolvido.",
        "Estou frustrado, {topic} deveria ter sido resolvido ontem.",
        "Não é a primeira vez que {topic} dá problema, isso está me incomodando.",
        "Poderiam ter avisado antes sobre {topic}, fiquei chateado com a demora.",
        "Esperava mais organização com {topic}, estou insatisfeito.",
        "{topic} de novo com problema, começa a ficar cansativo.",
        "Não gostei nada de como {topic} foi tratado, esperava mais respeito.",
        "Que chato, {topic} sempre trava e ninguém avisa antes.",
        "Confesso que fiquei decepcionado com {topic}, podia ser bem melhor.",
        "Achei o atendimento sobre {topic} meio desorganizado, fiquei incomodado.",
        "Poxa, {topic} de novo com demora, isso não devia acontecer.",
        "Não é grave, mas {topic} me deixou incomodado hoje.",
    ],
    0.0: [
        "Qual o procedimento para {topic}?",
        "Vocês podem me informar sobre {topic}?",
        "Preciso de detalhes sobre {topic}, pode ajudar?",
        "Bom dia, gostaria de confirmar um detalhe sobre {topic}.",
        "Onde encontro informações sobre {topic}?",
        "Só para entender melhor, como funciona {topic} aqui?",
        "Oi, tudo bem? Queria saber sobre {topic}.",
        "Boa tarde, alguém pode me explicar {topic}?",
        "Olá, só confirmando um detalhe sobre {topic}.",
        "Estou apenas verificando o status de {topic}.",
        "Bom dia, {topic} está previsto para qual horário?",
        "Oi, existe algum custo adicional relacionado a {topic}?",
    ],
    0.5: [
        "Obrigado, o problema com {topic} foi resolvido rapidinho!",
        "Adorei o atendimento sobre {topic}, muito rápido.",
        "Ficou tudo certo com {topic}, agradeço a atenção.",
        "Gostei de como cuidaram de {topic}, bem tranquilo.",
        "Fiquei satisfeito com {topic}, valeu pela ajuda!",
        "Show, {topic} funcionou direitinho, obrigado pelo suporte.",
        "Muito obrigado pela atenção com {topic}, ficou tudo certo!",
        "Legal, {topic} resolvido rapidinho, valeu!",
        "Gostei bastante de como trataram {topic}, nota boa!",
        "Bacana, {topic} ficou bem resolvido, agradeço a equipe.",
        "Tudo certo com {topic}, obrigado pela paciência!",
        "Fiquei contente com {topic}, deu tudo certo no final.",
    ],
    1.0: [
        "Vocês são incríveis! {topic} superou minhas expectativas, super recomendo!!",
        "Melhor experiência que já tive, o suporte pra {topic} foi perfeito!!",
        "Simplesmente maravilhoso como resolveram {topic}, voltarei com certeza!!",
        "Estou muito feliz, {topic} foi impecável, parabéns à equipe!!",
        "Nota 10 para {topic}, atendimento excelente, super indico!!",
        "Que atendimento sensacional sobre {topic}, amei tudo!!",
        "Amei demais como cuidaram de {topic}, vocês são demais!!",
        "Que atendimento excelente sobre {topic}, super indico a todos!!",
        "Simplesmente perfeito o jeito que resolveram {topic}, nota mil!!",
        "Adorei cada detalhe relacionado a {topic}, experiência incrível!!",
        "Excelente, {topic} foi maravilhoso, amei a estadia toda!!",
        "Uau, {topic} superou tudo, experiência incrível, recomendo demais!!",
        "Nota 10 para {topic}, atendimento excelente, super indico!!",
        "Que atendimento sensacional sobre {topic}, amei tudo!!",
    ],
}

_NEUTRAL_OPENERS = [
    "Oi, tudo bem?",
    "Bom dia, preciso de uma informação.",
    "Olá, posso fazer uma pergunta?",
    "Oi, estou com uma dúvida rápida.",
]


def _build_examples(seed: int = 42) -> list[tuple[list[dict], float]]:
    rng = random.Random(seed)
    examples: list[tuple[list[dict], float]] = []

    for label, templates in _TEMPLATES.items():
        for template in templates:
            for topic in _TOPICS:
                trigger_text = template.format(topic=topic)
                history = [{"role": "customer", "text": trigger_text}]
                # ~25% of examples carry a prior customer turn as context (ADR-0007/0008).
                if rng.random() < 0.25:
                    opener = rng.choice(_NEUTRAL_OPENERS)
                    history = [{"role": "customer", "text": opener}] + history
                examples.append((history, label))

    rng.shuffle(examples)
    return examples


def _code_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unknown"


def _make_pipeline() -> Pipeline:
    # ADR-0008 Approach A: word + char n-grams on the trigger message, word n-grams
    # on the concatenated context, combined and fed into Ridge.
    trigger_features = FeatureUnion(
        [
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("trigger", trigger_features, "text_clean"),
            ("context", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1), "context_text"),
        ]
    )
    return Pipeline([("features", preprocessor), ("ridge", Ridge(alpha=1.0))])


def train_model() -> None:
    examples = _build_examples()
    rows = [features_to_row(extract_features(history)) for history, _ in examples]
    labels = [label for _, label in examples]

    frame = pd.DataFrame(rows)
    X_train, X_test, y_train, y_test = train_test_split(frame, labels, test_size=0.2, random_state=42)

    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    pipeline = _make_pipeline().fit(X_train, y_train)

    for name, model in (("baseline", baseline), ("tfidf-ridge", pipeline)):
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        spearman = spearmanr(y_test, preds).correlation
        print(f"[{name}] mae={mae:.4f} rmse={rmse:.4f} spearman={spearman:.4f}")

    test_preds = pipeline.predict(X_test)
    metrics = {
        "mae": round(float(mean_absolute_error(y_test, test_preds)), 4),
        "rmse": round(float(mean_squared_error(y_test, test_preds) ** 0.5), 4),
        "spearman": round(float(spearmanr(y_test, test_preds).correlation), 4),
    }

    # Ship the version trained on the full dataset, not just the train split.
    final_pipeline = _make_pipeline().fit(frame, labels)

    now = datetime.now(timezone.utc)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    version_number = 0
    while True:
        model_version = f"mood-{now.strftime('%Y.%m')}.{version_number}"
        version_dir = MODELS_DIR / model_version
        try:
            version_dir.mkdir()
            break
        except FileExistsError:
            version_number += 1

    joblib.dump(final_pipeline, version_dir / "model.joblib")

    manifest = {
        "model_version": model_version,
        "scale": "-1 to 1",
        "mood_labels": None,
        "trained_at": now.isoformat().replace("+00:00", "Z"),
        "dataset_id": "ds-mvp-synthetic-inline",
        "label_set_id": "ls-mvp-synthetic-inline",
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "history_window": HISTORY_WINDOW,
        "history_scope": HISTORY_SCOPE,
        "algorithm": ALGORITHM,
        "metrics": metrics,
        "code_commit": _code_commit(),
    }
    (version_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    active = {"model_version": model_version, "promoted_at": now.isoformat().replace("+00:00", "Z")}
    (MODELS_DIR / "active.json").write_text(json.dumps(active, indent=2), encoding="utf-8")

    print(f"registered and promoted {model_version} ({len(examples)} examples)")


if __name__ == "__main__":
    train_model()
