"""
Custom colormaps converted from MATLAB definitions.

All colormaps are exactly as defined in the user's MATLAB fun_colormaps function.
The function extend_colormap() can be used to create versions with more colors
via linear interpolation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Store base colormap data exactly as defined in MATLAB

# TEMP2 colormap - User's preferred temperature colormap  
# From MATLAB: cm.TEMP2=flipud([165 0 38; 215 48 39; 244 109 67; 253 174 95; 254 224 144; 255 255 191; 224 243 248; 171 217 233; 116 173 209; 69 117 180; 49 54 149]/255);
TEMP2_DATA = np.array([
    [49/255,  54/255,  149/255],   # Deep blue
    [69/255,  117/255, 180/255],   # Blue  
    [116/255, 173/255, 209/255],   # Light blue
    [171/255, 217/255, 233/255],   # Very light blue
    [224/255, 243/255, 248/255],   # Pale blue
    [255/255, 255/255, 191/255],   # Pale yellow
    [254/255, 224/255, 144/255],   # Light yellow
    [253/255, 174/255, 95/255],    # Orange-yellow
    [244/255, 109/255, 67/255],    # Orange
    [215/255, 48/255,  39/255],    # Red-orange
    [165/255, 0/255,   38/255],    # Dark red
])

# SAL colormap - User's preferred salinity colormap
# From MATLAB: cm.SAL = flipud([[255 255 217]; [237 248 217]; [199 233 180]; [127 205 187]; [65 182 196]; [29 145 192]; [34 94 168]; [37 52 148]; [8 29 88]]/255);
SAL_DATA = np.array([
    [8/255,   29/255,  88/255],   # Deep blue
    [37/255,  52/255,  148/255],  # Blue
    [34/255,  94/255,  168/255],  # Medium blue
    [29/255,  145/255, 192/255],  # Light blue
    [65/255,  182/255, 196/255],  # Cyan-blue
    [127/255, 205/255, 187/255],  # Blue-green
    [199/255, 233/255, 180/255],  # Light green
    [237/255, 248/255, 217/255],  # Pale yellow-green
    [255/255, 255/255, 217/255],  # Pale yellow
])

# TEMP colormap - Original temperature colormap
# From MATLAB: cm.TEMP = flipud([[215 48 39]; [244 109 67]; [253 174 97]; [254 224 144]; [255 255 191]; [224 243 248]; [171 217 233]; [116 173 209]; [69 117 180]; [5 48 97]]/255);
TEMP_DATA = np.array([
    [5/255,   48/255,  97/255],    # Deep blue
    [69/255,  117/255, 180/255],   # Blue
    [116/255, 173/255, 209/255],   # Light blue
    [171/255, 217/255, 233/255],   # Very light blue
    [224/255, 243/255, 248/255],   # Pale blue
    [255/255, 255/255, 191/255],   # Pale yellow
    [254/255, 224/255, 144/255],   # Light yellow
    [253/255, 174/255, 97/255],    # Orange-yellow
    [244/255, 109/255, 67/255],    # Orange
    [215/255, 48/255,  39/255],    # Red
])

# TOPO colormap - For bathymetry/topography
# From MATLAB: cm.TOPO = [[84 48 5]; [140 81 10]; [191 129 45]; [246 232 195]; [245 245 245]; [199 234 229]; [128 205 193]; [53 151 143]; [1 102 94]; [0 60 48]]/255;
TOPO_DATA = np.array([
    [84/255,  48/255,  5/255],     # Dark brown
    [140/255, 81/255,  10/255],    # Brown
    [191/255, 129/255, 45/255],    # Light brown
    [246/255, 232/255, 195/255],   # Beige
    [245/255, 245/255, 245/255],   # White
    [199/255, 234/255, 229/255],   # Very light blue-green
    [128/255, 205/255, 193/255],   # Light blue-green
    [53/255,  151/255, 143/255],   # Blue-green
    [1/255,   102/255, 94/255],    # Dark blue-green
    [0/255,   60/255,  48/255],    # Very dark blue-green
])

# POLAR colormap - Red to blue through white (for anomalies)
# From MATLAB: cm.POLAR = flipud([[103 0 31]; [178 24 43]; [214 96 77]; [244 165 130]; [253 219 199]; [255 255 240]; [209 229 240]; [146 197 222]; [67 147 195]; [33 102 172]; [5 48 97]]/255);
POLAR_DATA = np.array([
    [5/255,   48/255,  97/255],    # Deep blue
    [33/255,  102/255, 172/255],   # Blue
    [67/255,  147/255, 195/255],   # Light blue
    [146/255, 197/255, 222/255],   # Very light blue
    [209/255, 229/255, 240/255],   # Pale blue
    [255/255, 255/255, 240/255],   # Off-white
    [253/255, 219/255, 199/255],   # Pale red
    [244/255, 165/255, 130/255],   # Light red
    [214/255, 96/255,  77/255],    # Red
    [178/255, 24/255,  43/255],    # Dark red
    [103/255, 0/255,   31/255],    # Very dark red
])

# OXY colormap - For oxygen
# From MATLAB: cm.OXY = ([[84,48,5]; [140,81,10]; [191,129,45]; [223,194,125]; [246,232,195]; [245,245,245]; [199,234,229]; [128,205,193]; [53,151,143]; [1,102,94]; [0,60,48]])/255;
OXY_DATA = np.array([
    [84/255,  48/255,  5/255],     # Dark brown
    [140/255, 81/255,  10/255],    # Brown
    [191/255, 129/255, 45/255],    # Light brown
    [223/255, 194/255, 125/255],   # Tan
    [246/255, 232/255, 195/255],   # Beige
    [245/255, 245/255, 245/255],   # White
    [199/255, 234/255, 229/255],   # Very light blue-green
    [128/255, 205/255, 193/255],   # Light blue-green
    [53/255,  151/255, 143/255],   # Blue-green
    [1/255,   102/255, 94/255],    # Dark blue-green
    [0/255,   60/255,  48/255],    # Very dark blue-green
])

# PurGre colormap - Purple to green (multipurpose, divergent)
# From MATLAB: cm.PurGre = flipud([[64 0 75]; [118 42 131]; [153 112 171]; [194 165 207]; [231 212 232]; [247 247 247]; [217 240 211]; [166 219 160]; [90 174 97]; [27 120 55]; [0 68 27]]/255);
PURGRE_DATA = np.array([
    [0/255,   68/255,  27/255],    # Dark green
    [27/255,  120/255, 55/255],    # Green
    [90/255,  174/255, 97/255],    # Light green
    [166/255, 219/255, 160/255],   # Very light green
    [217/255, 240/255, 211/255],   # Pale green
    [247/255, 247/255, 247/255],   # White
    [231/255, 212/255, 232/255],   # Pale purple
    [194/255, 165/255, 207/255],   # Light purple
    [153/255, 112/255, 171/255],   # Purple
    [118/255, 42/255,  131/255],   # Dark purple
    [64/255,  0/255,   75/255],    # Very dark purple
])

# CHL colormap - For chlorophyll (greens)
# From MATLAB: cm.CHL = ([[255 255 204]; [217 240 163]; [173 221 142]; [120 198 121]; [65 171 93]; [35 132 67]; [0 90 50]]/255);
CHL_DATA = np.array([
    [255/255, 255/255, 204/255],   # Pale yellow
    [217/255, 240/255, 163/255],   # Light yellow-green
    [173/255, 221/255, 142/255],   # Light green
    [120/255, 198/255, 121/255],   # Green
    [65/255,  171/255, 93/255],    # Medium green
    [35/255,  132/255, 67/255],    # Dark green
    [0/255,   90/255,  50/255],    # Very dark green
])

# Y2R colormap - Yellow to red
# From MATLAB: cm.Y2R=[[255 255 204]; [255 237 160]; [254 217 118]; [254 178 76]; [253 141 60]; [252 141 60]; [227 26 28]; [189 0 38]; [128 0 38]]/255;
Y2R_DATA = np.array([
    [255/255, 255/255, 204/255],   # Pale yellow
    [255/255, 237/255, 160/255],   # Yellow
    [254/255, 217/255, 118/255],   # Orange-yellow
    [254/255, 178/255, 76/255],    # Orange
    [253/255, 141/255, 60/255],    # Red-orange
    [252/255, 141/255, 60/255],    # Red-orange
    [227/255, 26/255,  28/255],    # Red
    [189/255, 0/255,   38/255],    # Dark red
    [128/255, 0/255,   38/255],    # Very dark red
])

# BLUE colormap - Blues to whites
# From MATLAB: cm.BLUE=[[255 247 251]; [236 231 242]; [208 209 230]; [166 189 219]; [116 169 207]; [54 144 192]; [5 112 176]; [4 90 141]; [2 56 88]]/255;
BLUE_DATA = np.array([
    [255/255, 247/255, 251/255],   # Almost white
    [236/255, 231/255, 242/255],   # Very pale purple-blue
    [208/255, 209/255, 230/255],   # Pale blue
    [166/255, 189/255, 219/255],   # Light blue
    [116/255, 169/255, 207/255],   # Medium blue
    [54/255,  144/255, 192/255],   # Blue
    [5/255,   112/255, 176/255],   # Dark blue
    [4/255,   90/255,  141/255],   # Very dark blue
    [2/255,   56/255,  88/255],    # Almost black blue
])

# CAT colormap - Categorical colors
# From MATLAB: cm.CAT=[[166 206 227]; [31 120 180]; [178 223 138]; [51 160 44]; [251 154 153]; [227 26 28]; [253 191 111]; [255 127 0]; [202 178 214]; [106 61 154]; [255 255 153]];
CAT_DATA = np.array([
    [166/255, 206/255, 227/255],   # Light blue
    [31/255,  120/255, 180/255],   # Blue
    [178/255, 223/255, 138/255],   # Light green
    [51/255,  160/255, 44/255],    # Green
    [251/255, 154/255, 153/255],   # Light red
    [227/255, 26/255,  28/255],    # Red
    [253/255, 191/255, 111/255],   # Orange
    [255/255, 127/255, 0/255],     # Dark orange
    [202/255, 178/255, 214/255],   # Light purple
    [106/255, 61/255,  154/255],   # Purple
    [255/255, 255/255, 153/255],   # Yellow
])



def extend_colormap(base_colors, n_colors):
    """
    Extend a colormap to have more colors via linear interpolation.
    
    Similar to MATLAB's fine_colormap function.
    
    Parameters
    ----------
    base_colors : np.ndarray
        Base colormap data (N x 3 RGB values)
    n_colors : int
        Desired number of colors in output
        
    Returns
    -------
    np.ndarray
        Extended colormap data (n_colors x 3 RGB values)
    """
    if n_colors <= len(base_colors):
        return base_colors
    
    # Create interpolation points
    original_points = np.linspace(0, 1, len(base_colors))
    new_points = np.linspace(0, 1, n_colors)
    
    # Interpolate each RGB channel
    extended_colors = np.zeros((n_colors, 3))
    for channel in range(3):
        extended_colors[:, channel] = np.interp(
            new_points, original_points, base_colors[:, channel]
        )
    
    return extended_colors


def create_custom_colormaps():
    """Create all custom colormaps and register them with matplotlib."""
    
    # Dictionary to store base colormap data
    base_data = {
        'TEMP2': TEMP2_DATA,
        'TEMP': TEMP_DATA,
        'SAL': SAL_DATA,
        'TOPO': TOPO_DATA,
        'POLAR': POLAR_DATA,
        'OXY': OXY_DATA,
        'PurGre': PURGRE_DATA,
        'CHL': CHL_DATA,
        'Y2R': Y2R_DATA,
        'BLUE': BLUE_DATA,
        'CAT': CAT_DATA,
    }
    
    colormaps = {}
    
    # Create LinearSegmentedColormap objects
    for name, data in base_data.items():
        colormaps[name] = LinearSegmentedColormap.from_list(name, data)
    
    # Register with matplotlib
    try:
        import matplotlib
        for name, cmap in colormaps.items():
            matplotlib.colormaps.register(name=name, cmap=cmap)
    except (AttributeError, ImportError):
        # Fallback for older matplotlib versions
        pass
    
    return colormaps

# Create and register the colormaps on import
CUSTOM_COLORMAPS = create_custom_colormaps()