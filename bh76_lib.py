"""
BH76 data: TURBOMOLE `coord` parser and the reaction / charge / spin tables.

Data sources (in data/):
  bh76_geoms.tar.gz                 -- GMTKN55 geometries, BH76/<species>/coord, TURBOMOLE format, Bohr
  bh76_reference_gmtkn55.csv        -- the 76 reactions + W2-F12 reference + GMTKN55's PBE deviation
  bh76_charge_multiplicity.txt      -- charge + multiplicity (2S+1) per species

Species names are case-sensitive: `C2H5` and `c2h5` are two different geometries (the HTBH38
ethyl radical and the NHTBH38 association-reaction ethyl radical). The geometries are read from
a tarball, so both names are kept on case-insensitive file systems (Windows, macOS).
"""
import os, tarfile

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = f"{ROOT}/data"
GEOMTAR = f"{DATA}/bh76_geoms.tar.gz"
BOHR = "Bohr"

_FILES = None


def _read(name, fname):
    """Text of BH76/<name>/<fname> from the geometry tarball, or None if absent."""
    global _FILES
    if _FILES is None:
        _FILES = {}
        with tarfile.open(GEOMTAR) as tf:
            for m in tf.getmembers():
                if m.isfile():
                    _FILES[m.name] = tf.extractfile(m).read().decode()
    return _FILES.get(f"BH76/{name}/{fname}")


# ---------------------------------------------------------------- geometry
def parse_coord(name):
    """TURBOMOLE $coord block -> PySCF atom list. Values are in Bohr."""
    text = _read(name, "coord")
    if text is None:
        raise FileNotFoundError(f"no geometry for BH76 species {name!r} in {GEOMTAR}")
    atoms, inblock = [], False
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("$coord"):
            inblock = True; continue
        if inblock:
            if s.startswith("$"):                 # $redundant / $user-defined / $end
                break
            p = s.split()
            if len(p) < 4:
                continue
            x, y, z, el = float(p[0]), float(p[1]), float(p[2]), p[3]
            atoms.append([el.capitalize(), (x, y, z)])   # 'cl' -> 'Cl'
    if not atoms:
        raise ValueError(f"empty $coord block for {name!r}")
    return atoms


def parse_xyz(name):
    """The struc.xyz shipped alongside coord, in Angstrom. Used only as a cross-check."""
    text = _read(name, "struc.xyz")
    if text is None:
        return None
    lines = [l for l in text.splitlines() if l.strip()]
    n = int(lines[0].split()[0])
    out = []
    for ln in lines[-n:]:
        p = ln.split()
        out.append([p[0], (float(p[1]), float(p[2]), float(p[3]))])
    return out


def uhf_marker(name):
    """The .UHF file GMTKN55 ships = number of unpaired electrons, or None if absent."""
    t = _read(name, ".UHF")
    if t is None:
        return None
    t = t.strip()
    return int(t) if t else None


# ---------------------------------------------------------------- tables
def load_species():
    """name -> (charge, multiplicity). Superset of what BH76 needs."""
    out = {}
    for ln in open(f"{DATA}/bh76_charge_multiplicity.txt"):
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        out[p[0]] = (int(p[1]), int(p[2]))
    return out


def load_reactions():
    """[(idx, [(species, coef), ...], ref_kcalmol, pbe_dev_nodisp), ...] -- 76 rows, in order,
    duplicates retained (rows 15/16, 19/20, 47/48, 75/76 are duplicates by construction)."""
    rxns = []
    for ln in open(f"{DATA}/bh76_reference_gmtkn55.csv"):
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("idx,"):
            continue
        p = s.split(",")
        idx = int(p[0])
        terms = [(p[1 + i], int(p[4 + i])) for i in range(3) if p[1 + i] and p[4 + i]]
        rxns.append((idx, terms, float(p[7]), float(p[8])))
    return rxns


def unique_species(rxns=None):
    """The species actually touched by the 76 reactions, in first-appearance order."""
    rxns = rxns or load_reactions()
    seen, out = set(), []
    for _, terms, _, _ in rxns:
        for nm, _c in terms:
            if nm not in seen:
                seen.add(nm); out.append(nm)
    return out


def build_mol(name, basis, verbose=0, species=None):
    """PySCF Mole for a BH76 species. Charge/spin come from the GMTKN55 table;
    geometry from the GMTKN55 coord file, in Bohr."""
    from pyscf import gto
    species = species or load_species()
    if name not in species:
        raise KeyError(f"{name!r} not in bh76_charge_multiplicity.txt")
    charge, mult = species[name]
    return gto.M(atom=parse_coord(name), unit=BOHR, basis=basis,
                 charge=charge, spin=mult - 1, verbose=verbose)


HARTREE2KCAL = 627.5095      # kcal/mol per Hartree, as used by GMTKN55
