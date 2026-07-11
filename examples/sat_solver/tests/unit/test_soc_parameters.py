"""Unit tests — SoC parameters and Wishbone address map correctness.

These tests assert purely in Python, with no simulator or hardware required.
They verify:
  - design constants match documented limits
  - byte-offset arithmetic in client.py is consistent with design.py word offsets
  - literal encoding packs and unpacks correctly
  - solve() input validation rejects out-of-range arguments
  - WishboneRequest wire serialisation matches the documented wire format
  - WishboneResponse deserialisation is the inverse of serialisation
  - parse_dimacs() handles standard DIMACS CNF files
  - solve() correctly decodes the model register into a variable assignment
"""

import importlib.util
import struct
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Load design and client from the example directory so tests are runnable from
# any working directory and do not depend on PYTHONPATH being set.
# ---------------------------------------------------------------------------
_EXAMPLE_DIR = Path(__file__).resolve().parents[2]  # examples/sat_solver/
sys.path.insert(0, str(_EXAMPLE_DIR))  # needed for client's "from design import ..."


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _EXAMPLE_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


design = _load("sat_solver_design", "design.py")

# client.py does a bare ``from design import ...``. When the whole examples/
# tree is collected in one pytest run, another example's design.py may already
# be cached in sys.modules under the name "design" (e.g. ffn_accel puts its dir
# on sys.path and imports ``design`` directly). Bind THIS example's design to
# that name only while client.py is being imported, then restore the previous
# value so we don't perturb the shared session.
_prev_design = sys.modules.get("design")
sys.modules["design"] = design
try:
    client = _load("sat_solver_client", "client.py")
finally:
    if _prev_design is not None:
        sys.modules["design"] = _prev_design
    else:
        del sys.modules["design"]


# ---------------------------------------------------------------------------
# 1. SoC / design parameter assertions
# ---------------------------------------------------------------------------


class TestDesignConstants:
    """Documented limits from README and design.py must be exact."""

    def test_max_vars(self):
        assert design.MAX_VARS == 10, "Hardware supports up to 10 boolean variables"

    def test_max_clauses(self):
        assert design.MAX_CLAUSES == 20, "Hardware supports up to 20 clauses"

    def test_clause_len(self):
        assert design.CLAUSE_LEN == 10, "Each clause holds up to 10 literals"

    def test_lit_base_word_offset(self):
        assert design.LIT_BASE == 8, "Literal block starts at word offset 8"

    def test_literal_register_count(self):
        assert design.MAX_CLAUSES * design.CLAUSE_LEN == 200, (
            "Total literal slots = MAX_CLAUSES * CLAUSE_LEN = 200"
        )

    def test_worst_case_cycles(self):
        """2^MAX_VARS assignments is the documented worst-case cycle count."""
        assert (1 << design.MAX_VARS) == 1024

    def test_client_imports_match_design(self):
        """client.py imports must resolve to the same values as design.py."""
        assert client.MAX_VARS == design.MAX_VARS
        assert client.MAX_CLAUSES == design.MAX_CLAUSES
        assert client.CLAUSE_LEN == design.CLAUSE_LEN
        assert client.LIT_BASE == design.LIT_BASE


# ---------------------------------------------------------------------------
# 2. Wishbone address map — byte offsets derived from word offsets
# ---------------------------------------------------------------------------


class TestWishboneAddressMap:
    """Byte offsets in client.py must equal 4 × the word offsets in design.py."""

    # Word-offset → (expected byte offset, client constant name, value)
    _MAP = [
        (0, "REG_CTRL",     0x00),
        (1, "REG_NVARS",    0x04),
        (2, "REG_NCLAUSES", 0x08),
        (3, "REG_MODEL",    0x0C),
        (8, "REG_LITERALS", 0x20),   # LIT_BASE * 4
    ]

    @pytest.mark.parametrize("word_off,name,expected_byte", _MAP)
    def test_byte_offset_formula(self, word_off, name, expected_byte):
        assert word_off * 4 == expected_byte, (
            f"{name}: 4×{word_off} should be {expected_byte:#04x}"
        )

    def test_reg_ctrl_value(self):
        assert client.REG_CTRL == 0x00

    def test_reg_nvars_value(self):
        assert client.REG_NVARS == 0x04

    def test_nclauses_burst_offset_matches_design_word_2(self):
        """n_clauses has no named constant; solve() writes it as the second
        word in the REG_NVARS burst. Verify this lands on design word 2."""
        nclauses_word_offset = 2  # design.py register map, word 2
        assert client.REG_NVARS + 4 == nclauses_word_offset * 4

    def test_reg_model_value(self):
        assert client.REG_MODEL == 0x0C

    def test_reg_literals_value(self):
        assert client.REG_LITERALS == 4 * design.LIT_BASE

    def test_reg_literals_no_overlap_with_control(self):
        """Literal block must start after the five control/status words."""
        first_control_word_past_status = 5  # words 0-4 are control/status
        assert design.LIT_BASE >= first_control_word_past_status

    def test_literal_block_end_address(self):
        """Last literal word offset must fit in the 9-bit address field (512 words)."""
        last_word = design.LIT_BASE + design.MAX_CLAUSES * design.CLAUSE_LEN - 1
        assert last_word < 512, f"Last literal at word {last_word} overflows 9-bit adr"

    def test_individual_literal_word_offsets(self):
        """Word offset for clause c, slot l = LIT_BASE + c*CLAUSE_LEN + l."""
        for c in range(design.MAX_CLAUSES):
            for slot in range(design.CLAUSE_LEN):
                expected = design.LIT_BASE + c * design.CLAUSE_LEN + slot
                assert expected < 512


# ---------------------------------------------------------------------------
# 3. Literal encoding
# ---------------------------------------------------------------------------


class TestEncodeLiterals:
    """encode_literals() must produce 200 words with correct bit packing."""

    def test_output_length(self):
        clauses = [[1]]
        words = client.encode_literals(clauses)
        assert len(words) == design.MAX_CLAUSES * design.CLAUSE_LEN

    def test_empty_clauses_all_zeros(self):
        """Unused slots must be zero (used-bit = 0)."""
        words = client.encode_literals([[1]])
        # slot 0 of clause 0 is used; all others must be 0
        assert words[1] == 0
        assert all(w == 0 for w in words[design.CLAUSE_LEN:])

    def test_positive_literal_encoding(self):
        """Positive literal k → var=k-1, neg=0, used=1."""
        words = client.encode_literals([[3]])
        w = words[0]
        var  = w & 0x0F        # bits[3:0]
        neg  = (w >> 4) & 0x1  # bit4
        used = (w >> 5) & 0x1  # bit5
        assert var  == 2,  "var for x3 is 2 (0-based)"
        assert neg  == 0,  "positive literal → neg=0"
        assert used == 1,  "slot is occupied"

    def test_negative_literal_encoding(self):
        """Negative literal -k → var=k-1, neg=1, used=1."""
        words = client.encode_literals([[-5]])
        w = words[0]
        var  = w & 0x0F
        neg  = (w >> 4) & 0x1
        used = (w >> 5) & 0x1
        assert var  == 4,  "var for x5 is 4 (0-based)"
        assert neg  == 1,  "negative literal → neg=1"
        assert used == 1

    def test_clause_slot_placement(self):
        """Clause c, slot l maps to index c*CLAUSE_LEN + l."""
        clauses = [[], [2, -3]]   # clause 0 empty, clause 1 has two literals
        words = client.encode_literals(clauses)
        base = 1 * design.CLAUSE_LEN      # clause index 1
        w0 = words[base + 0]
        w1 = words[base + 1]
        # literal 2 → var=1, neg=0
        assert w0 & 0x0F == 1 and (w0 >> 4) & 1 == 0 and (w0 >> 5) & 1 == 1
        # literal -3 → var=2, neg=1
        assert w1 & 0x0F == 2 and (w1 >> 4) & 1 == 1 and (w1 >> 5) & 1 == 1

    def test_max_variable_index(self):
        """Variable MAX_VARS (1-based) → 0-based index MAX_VARS-1 fits in 4 bits."""
        words = client.encode_literals([[design.MAX_VARS]])
        var = words[0] & 0x0F
        assert var == design.MAX_VARS - 1
        assert var < 16  # 4-bit field

    def test_multiple_clauses(self):
        """All clauses of a full formula are encoded without collision."""
        clauses = [[1, -2], [3, 4, -5]]
        words = client.encode_literals(clauses)
        # Clause 0, slot 0: x1 positive
        assert words[0] & 0x0F == 0 and (words[0] >> 4) & 1 == 0
        # Clause 0, slot 1: x2 negative
        assert words[1] & 0x0F == 1 and (words[1] >> 4) & 1 == 1
        # Clause 1, slot 0: x3 positive  (offset = CLAUSE_LEN)
        idx = design.CLAUSE_LEN
        assert words[idx] & 0x0F == 2 and (words[idx] >> 4) & 1 == 0


# ---------------------------------------------------------------------------
# 4. solve() input validation
# ---------------------------------------------------------------------------


class TestSolveValidation:
    """solve() must raise ValueError before touching the connection."""

    def _mock_conn(self):
        return MagicMock()

    def _sat_conn(self):
        """Mock connection that returns done+sat, then a model word."""
        conn = MagicMock()
        conn.read.side_effect = [[0b11], [0b1]]
        return conn

    def test_zero_vars_rejected(self):
        with pytest.raises(ValueError, match="n_vars"):
            client.solve(self._mock_conn(), 0, [[1]])

    def test_too_many_vars_rejected(self):
        with pytest.raises(ValueError, match="n_vars"):
            client.solve(self._mock_conn(), design.MAX_VARS + 1, [[1]])

    def test_max_vars_accepted(self):
        """MAX_VARS itself must not raise."""
        client.solve(self._sat_conn(), design.MAX_VARS, [[1]])

    def test_empty_clauses_rejected(self):
        with pytest.raises(ValueError, match="clauses"):
            client.solve(self._mock_conn(), 1, [])

    def test_too_many_clauses_rejected(self):
        extra = [[1]] * (design.MAX_CLAUSES + 1)
        with pytest.raises(ValueError, match="clauses"):
            client.solve(self._mock_conn(), 1, extra)

    def test_clause_too_long_rejected(self):
        """A clause exceeding CLAUSE_LEN literals must raise before any I/O."""
        long_clause = list(range(1, design.CLAUSE_LEN + 2))  # CLAUSE_LEN+1 lits
        with pytest.raises(ValueError, match="literals"):
            client.solve(self._mock_conn(), design.MAX_VARS, [long_clause])

    def test_max_clauses_accepted(self):
        clauses = [[1]] * design.MAX_CLAUSES
        client.solve(self._sat_conn(), 1, clauses)


# ---------------------------------------------------------------------------
# 5. WishboneRequest wire serialisation
# ---------------------------------------------------------------------------


class TestWishboneRequestSerialization:
    """to_bytes() must match the documented wire format exactly."""

    def test_write_header_opcode(self):
        req = client.WishboneRequest(
            op=client.WishboneOp.WRITE, address=0x20, data=[0xDEAD]
        )
        raw = req.to_bytes()
        assert raw[0] == 0x01  # WRITE opcode

    def test_read_header_opcode(self):
        req = client.WishboneRequest(
            op=client.WishboneOp.READ, address=0x00, data=[1]
        )
        raw = req.to_bytes()
        assert raw[0] == 0x02  # READ opcode

    def test_length_field_is_3_bytes_big_endian(self):
        """Bytes 1-3 hold the 24-bit big-endian word count."""
        data_words = [1, 2, 3]
        req = client.WishboneRequest(
            op=client.WishboneOp.WRITE, address=0x00, data=data_words
        )
        raw = req.to_bytes()
        length = struct.unpack(">I", b"\x00" + raw[1:4])[0]
        assert length == len(data_words)

    def test_address_field_bytes_4_to_7(self):
        """Bytes 4-7 carry the 32-bit big-endian Wishbone byte address."""
        addr = 0x0000_0020
        req = client.WishboneRequest(
            op=client.WishboneOp.WRITE, address=addr, data=[0]
        )
        raw = req.to_bytes()
        parsed_addr = struct.unpack(">I", raw[4:8])[0]
        assert parsed_addr == addr

    def test_data_words_big_endian_after_header(self):
        """Each data word is serialised as a 4-byte big-endian integer."""
        words = [0xCAFEBABE, 0x00000001]
        req = client.WishboneRequest(
            op=client.WishboneOp.WRITE, address=0x00, data=words
        )
        raw = req.to_bytes()
        parsed = list(struct.unpack(f">{len(words)}I", raw[8:]))
        assert parsed == words

    def test_empty_data_produces_8_byte_header_only(self):
        req = client.WishboneRequest(
            op=client.WishboneOp.READ, address=0x00, data=[]
        )
        raw = req.to_bytes()
        assert len(raw) == 8  # 1 (op) + 3 (len) + 4 (addr)

    def test_total_length_is_8_plus_4_per_word(self):
        n = 5
        req = client.WishboneRequest(
            op=client.WishboneOp.WRITE, address=0x00, data=list(range(n))
        )
        raw = req.to_bytes()
        assert len(raw) == 8 + 4 * n

    def test_write_opcode_enum_value(self):
        assert int(client.WishboneOp.WRITE) == 0x01

    def test_read_opcode_enum_value(self):
        assert int(client.WishboneOp.READ) == 0x02

    def test_ctrl_start_burst_serialises_one_word(self):
        """The start pulse is a single-word write of value 1 to REG_CTRL."""
        req = client.WishboneRequest(
            op=client.WishboneOp.WRITE, address=client.REG_CTRL, data=[1]
        )
        raw = req.to_bytes()
        length = struct.unpack(">I", b"\x00" + raw[1:4])[0]
        assert length == 1
        word = struct.unpack(">I", raw[8:12])[0]
        assert word == 1


# ---------------------------------------------------------------------------
# 6. WishboneResponse status constants
# ---------------------------------------------------------------------------


class TestResponseStatus:
    def test_ok_is_zero(self):
        assert int(client.ResponseStatus.OK) == 0x00

    def test_error_is_one(self):
        assert int(client.ResponseStatus.ERROR) == 0x01

    def test_ok_property_true_when_status_ok(self):
        resp = client.WishboneResponse(
            status=client.ResponseStatus.OK, data=[0xAB]
        )
        assert resp.ok is True

    def test_ok_property_false_when_status_error(self):
        resp = client.WishboneResponse(
            status=client.ResponseStatus.ERROR, data=[]
        )
        assert resp.ok is False


# ---------------------------------------------------------------------------
# 7. parse_dimacs() — CNF file parsing
# ---------------------------------------------------------------------------


class TestParseDimacs:
    """parse_dimacs() must correctly handle standard DIMACS CNF syntax."""

    def _write_cnf(self, tmp_path, lines: list[str]) -> str:
        p = tmp_path / "test.cnf"
        p.write_text("\n".join(lines))
        return str(p)

    def test_simple_sat_formula(self, tmp_path):
        path = self._write_cnf(tmp_path, [
            "c comment line",
            "p cnf 3 2",
            "1 -2 0",
            "-1 3 0",
        ])
        n_vars, clauses = client.parse_dimacs(path)
        assert n_vars == 3
        assert clauses == [[1, -2], [-1, 3]]

    def test_header_sets_n_vars(self, tmp_path):
        path = self._write_cnf(tmp_path, ["p cnf 7 1", "1 0"])
        n_vars, _ = client.parse_dimacs(path)
        assert n_vars == 7

    def test_trailing_zero_stripped(self, tmp_path):
        path = self._write_cnf(tmp_path, ["p cnf 2 1", "1 2 0"])
        _, clauses = client.parse_dimacs(path)
        assert clauses == [[1, 2]]

    def test_comment_lines_ignored(self, tmp_path):
        path = self._write_cnf(tmp_path, [
            "c This is a comment",
            "c another comment",
            "p cnf 1 1",
            "c mid-file comment",
            "1 0",
        ])
        n_vars, clauses = client.parse_dimacs(path)
        assert n_vars == 1
        assert clauses == [[1]]

    def test_empty_lines_ignored(self, tmp_path):
        path = self._write_cnf(tmp_path, ["p cnf 1 1", "", "  ", "1 0"])
        _, clauses = client.parse_dimacs(path)
        assert clauses == [[1]]

    def test_negative_literals_preserved(self, tmp_path):
        path = self._write_cnf(tmp_path, ["p cnf 3 1", "-1 -2 -3 0"])
        _, clauses = client.parse_dimacs(path)
        assert clauses == [[-1, -2, -3]]

    def test_multiple_clauses(self, tmp_path):
        path = self._write_cnf(tmp_path, [
            "p cnf 4 3",
            "1 2 0",
            "-1 3 0",
            "-2 -3 4 0",
        ])
        _, clauses = client.parse_dimacs(path)
        assert len(clauses) == 3
        assert clauses[2] == [-2, -3, 4]


# ---------------------------------------------------------------------------
# 8. solve() model decoding — exercises production code via mock connection
# ---------------------------------------------------------------------------


class TestSolveModelDecoding:
    """solve() must decode the model register bit fields into variable assignments.

    These tests exercise the actual production code in client.solve(), not a
    local helper -- the mock drives conn.read() to return specific bit patterns
    and we assert on the returned assignment dict.
    """

    def _run_sat(self, n_vars: int, model_word: int) -> dict:
        """Run solve() with a mock that reports SAT and returns model_word."""
        conn = MagicMock()
        # First read: ctrl reg with done=1, sat=1 (0b11)
        # Second read: model register
        conn.read.side_effect = [[0b11], [model_word]]
        return client.solve(conn, n_vars, [[1]])

    def test_all_false_model(self):
        result = self._run_sat(4, 0b0000)
        assert result["result"] == "SAT"
        assert all(not v for v in result["assignment"].values())

    def test_all_true_model(self):
        result = self._run_sat(4, 0b1111)
        assert all(result["assignment"].values())

    def test_mixed_assignment(self):
        # 0b1010 → x1=F, x2=T, x3=F, x4=T
        result = self._run_sat(4, 0b1010)
        assert result["assignment"] == {
            "x1": False, "x2": True, "x3": False, "x4": True
        }

    def test_variable_count_respected(self):
        """Only n_vars variables appear in the assignment, regardless of model width."""
        result = self._run_sat(3, 0xFF)
        assert set(result["assignment"].keys()) == {"x1", "x2", "x3"}

    def test_bit0_is_x1(self):
        result = self._run_sat(4, 0b0001)
        assert result["assignment"]["x1"] is True
        assert result["assignment"]["x2"] is False

    def test_max_vars_model_width(self):
        """Model register is MAX_VARS bits wide; all bits must decode cleanly."""
        full_model = (1 << design.MAX_VARS) - 1
        result = self._run_sat(design.MAX_VARS, full_model)
        assert len(result["assignment"]) == design.MAX_VARS
        assert all(result["assignment"].values())

    def test_unsat_result_has_no_assignment(self):
        """When the solver reports UNSAT, solve() must not return an assignment key."""
        conn = MagicMock()
        conn.read.side_effect = [[0b01]]  # done=1, sat=0
        result = client.solve(conn, 1, [[1]])
        assert result["result"] == "UNSAT"
        assert "assignment" not in result
