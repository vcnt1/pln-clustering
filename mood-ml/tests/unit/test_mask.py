import pytest

from transform.mask import count_pii, mask_pii

CASES = [
    ("Meu e-mail e joao.silva@example.com, pode confirmar?", "<EMAIL>", "email"),
    ("Meu CPF e 480.453.698-27, por favor.", "<CPF>", "cpf"),
    ("Pode me ligar no (85) 91432-5568 amanha?", "<TEL>", "phone"),
    ("Meu numero e +55 11 91234-5678.", "<TEL>", "phone"),
    ("Ligar no 3234-5678 hoje.", "<TEL>", "phone"),
    ("Segue o numero bruto 11991234567 para contato.", "<TEL>", "phone"),
]


@pytest.mark.parametrize("text,marker,kind", CASES)
def test_mask_pii_replaces_each_kind(text: str, marker: str, kind: str) -> None:
    masked = mask_pii(text)
    assert marker in masked
    counts = count_pii(text)
    assert counts[kind] >= 1


def test_mask_pii_leaves_text_without_pii_untouched() -> None:
    text = "Obrigado pela ajuda, ficou tudo certo."
    assert mask_pii(text) == text
    assert count_pii(text) == {"cpf": 0, "phone": 0, "email": 0}


def test_mask_pii_is_idempotent() -> None:
    text = "Meu CPF e 480.453.698-27, meu email e a@example.com e meu tel (11) 91234-5678."
    once = mask_pii(text)
    twice = mask_pii(once)
    assert once == twice


def test_mask_pii_handles_all_three_kinds_in_one_message() -> None:
    text = "CPF 480.453.698-27, email a@example.com, tel (21) 92577-2724."
    masked = mask_pii(text)
    assert "<CPF>" in masked
    assert "<EMAIL>" in masked
    assert "<TEL>" in masked
    counts = count_pii(text)
    assert counts == {"cpf": 1, "phone": 1, "email": 1}


def test_count_pii_matches_what_mask_pii_actually_masks() -> None:
    text = "Contato: (11) 91234-5678 ou joao@example.org, CPF 111.222.333-44."
    counts = count_pii(text)
    masked = mask_pii(text)
    assert masked.count("<EMAIL>") == counts["email"]
    assert masked.count("<CPF>") == counts["cpf"]
    assert masked.count("<TEL>") == counts["phone"]
