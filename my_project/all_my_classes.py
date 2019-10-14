from my_project.colors                import *
from my_project.complex_numbers       import *
from my_project.magnetic_force        import *
from my_project.my_playground         import *
from my_project.my_plot_function      import *
from my_project.temperature_equations import *
from my_project.temperature           import *

OUTPUT_DIRECTORY = "all_my_classes_test"
SCENES_IN_ORDER = [
    ## colors
    Colors,

    ## complex_numbers
    StretchNumberLine,
    StretchNumberLine_example_0,
    StretchNumberLine_example_2,
    StretchNumberLine_example_1_5,
    StretchNumberLine_example_3,
    RotateNumberLine,
    RotateNumberLine_Twice,
    StretchNumberLine_summary,
    StretchNumberLine_summary_negative,
    StretchNumberLine_3_times,
    RotateComplexPlane,
    RotateComplexPlane_example_1,
    RotateComplexPlane_example_2,
    RotateComplexPlane_example_3,
    RotateComplexPlane_example_4,
    RotateComplexPlane_by_root_of_2,
    RotateComplexPlane_by_root_of_2_example,
    RotateComplexPlane_by_root_of_2_example_2_3_2,
    RotateComplexPlane_by_root_of_2_example_2_4_1,
    RotateComplexPlane_by_root_of_2_example_1_4_1,
    RotateComplexPlane_by_root_of_2_example_3_6_1,

    ## magnetic_force
    ReferenceFrame,
    ReferenceFrame_ProtonsRestFrame,
    ReferenceFrame_ProtonsRestFrame_WithForce,
    ReferenceFrame_ChargeRestFrame,
    ReferenceFrame_ChargeRestFrame_WithForce,
    ReferenceFrame_ChargeRestFrame_Appendix_1,
    ReferenceFrame_ChargeRestFrame_Appendix_2,
    ReferenceFrame_ChargeRestFrame_Appendix_3,
    ReferenceFrame_ChargeRestFrame_Appendix_4,
    ReferenceFrameTransform,
    ReferenceFrameIntroduction,
    ReferenceFrameIntroduction_A,
    ReferenceFrameIntroduction_B,
    ReferenceFrameIntroduction_COM,

    ## my_playground
    PlotParabola,
    PlotParabolaUpDown,
    PlotParabolaLeftRight,
    VisualizeStates,
    IntroduceVectorField,
    PlotFunctions,

    ## my_plot_function
    VectorFieldParabola,
    VectorFieldParabolaUpDown,
    VectorFieldParabolaLeftRight,
    MyParameterizedCurve,

    ## temperature
    IntroduceGasParticles,
    CollideToGas,
    MeltIce,
    MeltIce_For_Tests,
    MeltIce_Wide_Slow,
    MeltIce_Wide_Fast,
    MeltIce_Narrow_Slow,
    Ice,
    Ice_ExampleSmall,
    Ice_Examplelarge1,
    Ice_Examplelarge2,
    ShootParticles,
    ShootParticles_OneByOne,
    ShootParticles_OneByOne_Quick,
    ShootParticles_Together,
    ShootParticles_Together_Full,
    ShootParticles_Together_VeryFull,
    ShootParticles_Together_Full_NoForce2,
    ShootParticles_Together_Full_NoForce,
    ShootParticles_Together_VeryFull_NoForce,
    ShootParticles_Extruder,
    ShootParticles_test,
    Box,
    Box_Large,
    Box_Large_Dense,
    Box_Large_VeryDense,
    Box_Large_Light,
    Box_Large_Dense_Fast,
    Box_Small,
    RaiseTemperature,
    # IntroduceVectorField,
    # ShowFlow,

    ## temperature_equations
    Algebra,
    Algebra_test,
    LorentzForce,
]
