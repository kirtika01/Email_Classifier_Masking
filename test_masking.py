import unittest
from utils import mask_pii, validate_credit_card, validate_cvv, validate_expiry

class TestPIIMasking(unittest.TestCase):
    def test_full_name_masking(self):
        text = "My name is John Smith and I work at Tech Corp."
        masked_text, entities = mask_pii(text)
        self.assertIn("[full_name]", masked_text)
        self.assertTrue(any(e['classification'] == 'full_name' for e in entities))

    def test_email_masking(self):
        text = "Contact me at john.smith@example.com for details."
        masked_text, entities = mask_pii(text)
        self.assertIn("[email]", masked_text)
        self.assertTrue(any(e['classification'] == 'email' for e in entities))

    def test_phone_number_masking(self):
        text = "Call me at +1-123-456-7890 or 987.654.3210"
        masked_text, entities = mask_pii(text)
        self.assertIn("[phone_number]", masked_text)
        self.assertTrue(any(e['classification'] == 'phone_number' for e in entities))

    def test_dob_masking(self):
        text = "DOB: 15/03/1990 and License: ABC123"
        masked_text, entities = mask_pii(text)
        self.assertIn("[dob]", masked_text)
        self.assertTrue(any(e['classification'] == 'dob' for e in entities))

    def test_aadhar_masking(self):
        text = "Aadhar number: 1234 5678 9012"
        masked_text, entities = mask_pii(text)
        self.assertIn("[aadhar_num]", masked_text)
        self.assertTrue(any(e['classification'] == 'aadhar_num' for e in entities))

    def test_credit_card_masking(self):
        text = "Card: 4532-1234-5678-9012 Exp: 12/25 CVV: 123"
        masked_text, entities = mask_pii(text)
        self.assertIn("[credit_debit_no:****9012]", masked_text)
        self.assertIn("[expiry_no:**/**]", masked_text)
        self.assertIn("[cvv_no:***]", masked_text)

    def test_multiple_entities_masking(self):
        text = """
        Customer Details:
        Name: John Smith
        Email: john.smith@example.com
        Phone: +1-123-456-7890
        DOB: 15/03/1990
        Aadhar: 1234 5678 9012
        Card: 4532-1234-5678-9012
        Expiry: 12/25
        CVV: 123
        """
        masked_text, entities = mask_pii(text)
        
        expected_entities = [
            'full_name', 'email', 'phone_number', 'dob', 
            'aadhar_num', 'credit_debit_no', 'expiry_no', 'cvv_no'
        ]
        
        for entity_type in expected_entities:
            self.assertTrue(
                any(e['classification'] == entity_type for e in entities),
                f"Missing entity type: {entity_type}"
            )

class TestCardValidation(unittest.TestCase):
    def test_valid_credit_card(self):
        # Valid card numbers
        valid_cards = [
            "4532015112830366",  # Visa
            "4532 0151 1283 0366",  # Visa with spaces
            "4532-0151-1283-0366",  # Visa with hyphens
        ]
        for card in valid_cards:
            self.assertTrue(validate_credit_card(card))

    def test_invalid_credit_card(self):
        # Invalid card numbers
        invalid_cards = [
            "1234567890123",  # Too short
            "12345678901234567890",  # Too long
            "4532015112830367",  # Invalid checksum
        ]
        for card in invalid_cards:
            self.assertFalse(validate_credit_card(card))

    def test_valid_cvv(self):
        valid_cvvs = ["123", "1234"]
        for cvv in valid_cvvs:
            self.assertTrue(validate_cvv(cvv))

    def test_invalid_cvv(self):
        invalid_cvvs = ["12", "12345", "abc"]
        for cvv in invalid_cvvs:
            self.assertFalse(validate_cvv(cvv))

    def test_valid_expiry(self):
        valid_expiry = ["01/25", "12/29", "0125", "1229"]
        for exp in valid_expiry:
            self.assertTrue(validate_expiry(exp))

    def test_invalid_expiry(self):
        invalid_expiry = ["00/25", "13/29", "1/25", "0025"]
        for exp in invalid_expiry:
            self.assertFalse(validate_expiry(exp))

if __name__ == '__main__':
    unittest.main()