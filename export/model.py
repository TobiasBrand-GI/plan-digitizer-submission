import uuid
from enum import IntEnum
from typing import ClassVar

from osgeo import ogr
from pydantic import BaseModel, ConfigDict, Field


def new_gml_id():
    """Returns a fresh, unique gml:id value."""
    return f"GML_{uuid.uuid4()}"


class BesondereArtDerBaulNutzung(IntEnum):
    """XP_BesondereArtDerBaulNutzung -- member names are the code list's
    "kuerzel", so a parsed abbreviation maps via BesondereArtDerBaulNutzung[abbr].
    3000 (Wochenendhausgebiet) and 4000 (Sondergebiet, BauNVO 1962/68)
    share the kuerzel "SO" with 2100 and are left out."""
    WS = 1000
    WR = 1100
    WA = 1200
    WB = 1300
    MD = 1400
    MDW = 1450
    MI = 1500
    MU = 1550
    MK = 1600
    GE = 1700
    GI = 1800
    SO_E = 2000  # kuerzel "SO E"
    SO = 2100
    SG = 9999


class AllgArtDerBaulNutzung(IntEnum):
    """XP_AllgArtDerBaulNutzung"""
    W = 1000  # WohnBauflaeche
    M = 2000  # GemischteBauflaeche
    G = 3000  # GewerblicheBauflaeche
    S = 4000  # SonderBauflaeche
    B = 9999  # SonstigeBauflaeche


# §1 Abs. 2 BauNVO: which Baugebiet belongs to which Bauflaeche
_ALLG_FROM_BESONDERE = {
    **dict.fromkeys(["WS", "WR", "WA", "WB"], AllgArtDerBaulNutzung.W),
    **dict.fromkeys(["MD", "MDW", "MI", "MU", "MK"], AllgArtDerBaulNutzung.M),
    **dict.fromkeys(["GE", "GI"], AllgArtDerBaulNutzung.G),
    **dict.fromkeys(["SO_E", "SO"], AllgArtDerBaulNutzung.S),
    "SG": AllgArtDerBaulNutzung.B,
}


class ArtHoehenbezugspunkt(IntEnum):
    """XP_ArtHoehenbezugspunkt"""
    TH = 1000
    FH = 2000
    OK = 3000
    LH = 3500
    SH = 4000
    EFH = 4500
    HBA = 5000
    UK = 5500
    GBH = 6000
    WH = 6500
    GOK = 6600


class Rechtscharakter(IntEnum):
    """XP_Rechtscharakter -- only the values a BPlan export can use."""
    FestsetzungBPlan = 1000
    NachrichtlicheUebernahme = 2000
    Hinweis = 6000
    Kennzeichnung = 7000
    Unbekannt = 9998


class BPPlanArt(IntEnum):
    """BP_PlanArt -- subset."""
    BPlan = 1000
    EinfacherBPlan = 10000
    QualifizierterBPlan = 10001
    VorhabenbezogenerBPlan = 3000
    Sonstiges = 9999


class _GeometryModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Gemeinde(BaseModel):
    """XP_Gemeinde -- all properties optional in the XSD."""
    ags: str | None = None
    gemeindeName: str | None = None


class Hoehenangabe(BaseModel):
    """XP_Hoehenangabe; values in metres."""
    bezugspunkt: ArtHoehenbezugspunkt
    hMin: float | None = None
    hMax: float | None = None
    hZwingend: float | None = None
    h: float | None = None


class Dachform(IntEnum):
    """BP_Dachform -- member names are the code list's "kuerzel"."""
    FD = 1000   # Flachdach
    PD = 2100   # Pultdach
    VPD = 2200  # VersetztesPultdach
    GD = 3000   # GeneigtesDach
    SD = 3100   # Satteldach
    WD = 3200   # Walmdach
    KWD = 3300  # Krueppelwalmdach
    MD = 3400   # Mansarddach
    ZD = 3500   # Zeltdach
    KeD = 3600  # Kegeldach
    KuD = 3700  # Kuppeldach
    ShD = 3800  # Sheddach
    BD = 3900   # Bogendach
    TuD = 4000  # Turmdach
    ToD = 4100  # Tonnendach
    GDF = 5000  # Mischform
    SDF = 9999  # Sonstiges


class Dachgestaltung(BaseModel):
    """BP_Dachgestaltung; roof pitch in degrees."""
    dachform: list[Dachform] = []
    DNmin: float | None = None
    DNmax: float | None = None
    DN: float | None = None
    DNzwingend: float | None = None


class BaugebietsTeilFlaeche(_GeometryModel):
    flaechenschluss: ClassVar[bool] = True

    gml_id: str = Field(default_factory=new_gml_id)
    position: ogr.Geometry
    rechtscharakter: Rechtscharakter = Rechtscharakter.FestsetzungBPlan
    hoehenangabe: list[Hoehenangabe] = []
    dachgestaltung: list[Dachgestaltung] = []
    GFZ: float | None = None
    GRZ: float | None = None
    Zmin: int | None = None
    Zmax: int | None = None
    Zzwingend: int | None = None
    Z: int | None = None
    besondereArtDerBaulNutzung: BesondereArtDerBaulNutzung | None = None

    @property
    def allgArtDerBaulNutzung(self):
        """Derives the general use category (XP_AllgArtDerBaulNutzung) from
        besondereArtDerBaulNutzung per §1 Abs. 2 BauNVO; None if unset."""
        if self.besondereArtDerBaulNutzung is None:
            return None
        return _ALLG_FROM_BESONDERE[self.besondereArtDerBaulNutzung.name]


class UeberbaubareGrundstuecksFlaeche(_GeometryModel):
    """Building window (red outline). Geometry only for now: the XSD
    forbids setting the same attribute on both this overlay and the
    BaugebietsTeilFlaeche below it, and the text attributes go there."""
    flaechenschluss: ClassVar[bool] = False

    gml_id: str = Field(default_factory=new_gml_id)
    position: ogr.Geometry
    rechtscharakter: Rechtscharakter = Rechtscharakter.FestsetzungBPlan


class Bereich(_GeometryModel):
    gml_id: str = Field(default_factory=new_gml_id)
    nummer: int = 0
    geltungsbereich: ogr.Geometry | None = None


class Plan(_GeometryModel):
    gml_id: str = Field(default_factory=new_gml_id)
    name: str
    raeumlicherGeltungsbereich: ogr.Geometry
    gemeinde: list[Gemeinde] = Field(min_length=1)
    planArt: list[BPPlanArt] = Field(default=[BPPlanArt.BPlan], min_length=1)


class XPlanAuszug(BaseModel):
    """Everything one export writes -- one plan with one Bereich."""
    gml_id: str = Field(default_factory=new_gml_id)
    plan: Plan
    bereich: Bereich
    baugebiete: list[BaugebietsTeilFlaeche] = []
    baufenster: list[UeberbaubareGrundstuecksFlaeche] = []
