from __future__ import division, print_function, unicode_literals
__copyright__ = 'Copyright (C) Volker Diels-Grabsch <v@njh.eu>'

from collections import namedtuple
from pyomo.environ import ConcreteModel
from pyomo.environ import ConstraintList
from pyomo.environ import Objective
from pyomo.environ import Reals
from pyomo.environ import Set
from pyomo.environ import Var
from pyomo.environ import exp
from pyomo.environ import log
from pyomo.environ import maximize
from pyomo.environ import minimize
from pyomo.environ import value
from pyomo.opt import SolverFactory
from random import Random

Area = namedtuple('Area', 'max_x, max_y')
Geometry = namedtuple('Geometry', 'polygon')
Graph = namedtuple('Graph', 'points, segments')
Innergroup = namedtuple('Innergroup', 'geometries')
InnergroupPlacement = namedtuple('InnergroupPlacement', 'c, s, x, y')
Inputdata = namedtuple('Inputdata', 'area, outergroups')
Outergroup = namedtuple('Outergroup', 'innergroups')
OutergroupPlacement = namedtuple('OutergroupPlacement', 'innergroup_placements')
Placement = namedtuple('Placement', 'outergroup_placements')
Point = namedtuple('Point', 'x, y')
Segment = namedtuple('Segment', 'i, k')

SAMPLE_INPUT_DATA = Inputdata(
    area=Area(
        max_x=2000.0,
        max_y=1000.0,
    ),
    outergroups=[
        Outergroup(innergroups=[
            Innergroup(geometries=[
                Geometry(polygon=[
                    Point(x=0.0, y=0.0),
                    Point(x=size, y=size),
                    Point(x=0.0, y=size),
                ]),
                Geometry(polygon=[
                    Point(x=0.0, y=0.0),
                    Point(x=size*2, y=size*2),
                    Point(x=0.0, y=size*2),
                ])
            ])
            for size in [100.0, 100.0, 100.0, 200.0, 300.0, 400.0, 500.0, 500.0]
        ]),
        Outergroup(innergroups=[
            Innergroup(geometries=[
                Geometry(polygon=[
                    Point(x=0.0, y=0.0),
                    Point(x=0.0, y=size),
                    Point(x=size, y=0.0),
                ])
            ])
            for size in [250.0, 350.0, 350.0, 350.0, 350.0, 450.0, 550.0]
        ]),
        Outergroup(innergroups=[
            Innergroup(geometries=[
                Geometry(polygon=[
                    Point(x=0.0, y=0.0),
                    Point(x=size_x, y=0.0),
                    Point(x=size_x, y=size_y),
                    Point(x=0.0, y=size_y),
                ])
            ])
            for size_x, size_y in [(150.0, 150.0), (150.0, 450.0), (550.0, 550.0)]
        ]),
    ],
)

def remove_duplicates_keep_order(l):
    seen = set()
    result = []
    for e in l:
        if not e in seen:
            seen.add(e)
            result.append(e)
    return result

def polygons_to_graph(polygons):
    points = []
    point_index = {}
    for polygon in polygons:
        for point in polygon:
            if point not in point_index:
                point_index[point] = len(points)
                points.append(point)
    segments = remove_duplicates_keep_order(
        Segment(*sorted([point_index[polygon[i-1]], point_index[polygon[i]]]))
        for polygon in polygons
        for i in xrange(len(polygon))
    )
    return Graph(
        points=points,
        segments=segments,
    )

def innergroup_to_graph(innergroup):
    return polygons_to_graph([geometry.polygon for geometry in innergroup.geometries])

def inputdata_to_graphs(inputdata):
    return {
        (outergroup_i, innergroup_i): innergroup_to_graph(innergroup)
        for outergroup_i, outergroup in enumerate(inputdata.outergroups)
        for innergroup_i, innergroup in enumerate(outergroup.innergroups)
    }

def random_coordinate_offset(random_gen, input_min, input_max, output_min, output_max):
    offset_min = output_min - input_min
    offset_max = output_max - input_max
    if offset_min == offset_max:
        return offset_min
    if offset_min > offset_max:
        raise ValueError('Unable to move coordinates from [{input_min},{input_max}] (length {offset_min}) to [{output_min},{output_max}] (length {offset_max})'.format(**locals()))
    return random_gen.uniform(offset_min, offset_max)

def random_innergroup_placement(random_gen, innergroup, area):
    points = set(point for geometry in innergroup.geometries for point in geometry.polygon)
    min_x = min(point.x for point in points)
    max_x = max(point.x for point in points)
    min_y = min(point.y for point in points)
    max_y = max(point.y for point in points)
    return InnergroupPlacement(
        c=1.0,
        s=0.0,
        x=random_coordinate_offset(random_gen, min_x, max_x, 0, area.max_x),
        y=random_coordinate_offset(random_gen, min_y, max_y, 0, area.max_y),
    )

def random_placement(random_gen, inputdata):
    return Placement(
        outergroup_placements=[
            OutergroupPlacement(
                innergroup_placements=[
                    random_innergroup_placement(random_gen, innergroup, inputdata.area)
                    for innergroup in outergroup.innergroups
                ]
            )
            for outergroup in inputdata.outergroups
        ],
    )

def identity_placement(inputdata):
    return Placement(
        outergroup_placements=[
            OutergroupPlacement(
                innergroup_placements=[
                    Placement(
                        c=1.0,
                        s=0.0,
                        x=0.0,
                        y=0.0,
                    )
                    for innergroup in outergroup.innergroups
                ]
            )
            for outergroup in inputdata.outergroups
        ],
    )

def cap_smooth(x, stretch):
    '''Smooth variant of min(max(x, 0), 1)

    This is based on the inverse of:

        f(x) = 2 * (0-x)/((1-x)*(-1-x))

    which has two poles at x = 1 and x = -1.  The factor -x makes the function
    grow from -inf to +inf, rather than +inf to f(0) back to +inf.  Finally,
    the factor 2 is technically not needed, but simplifies the inverse.

    The inverse of f(x) is:

        g(x) = (sqrt(x**2 + 1) - 1)/x

    So g(x) is a "smooth jump" from y = -1 to y = 1 at the point x = 0.
    However, we need a jump from 0 to 1, so we define a helper function which
    moves and scales as needed:

        h(x) = (g(x)+1)/2

    We can make the jump "sharper" by scaling x with a factor s (e.g. s=100),
    leading to the final jump function:

        j(s, x) = h(s*x)

    This is just one of many possible approximations to the Heaviside
    step function. For alternatives, see:

        https://en.wikipedia.org/wiki/Heaviside_step_function#Analytic_approximations

    A single smooth edge at x = 0 is then given by:

        e(s, x) = j(s, x)*x

    Note that this final multiplication with x cancels out the division by x
    that was needed in g(x).  So e(s, x) gets along without division, but
    still needs sqrt:

        e(s, x) = x/2 + (sqrt(s**2 * x**2 + 1) - 1)/(2*s)

    The final function is given by overlaying two smooth edges:

        e(s, x) - e(s, x-1)

    That's it.
    '''
    def e(s, x):
        return x/2 + ((s**2 * x**2 + 1)**(1/2) - 1)/(2*s)
    s = stretch
    return e(s, x) - e(s, x-1)

def dist2_point_segment(px, py, ax, ay, bx, by, seg_len2):
    '''Squared distance (dist^2) of point p to segment ab'''
    # See https://math.stackexchange.com/a/330329
    dx = bx-ax
    dy = by-ay
    ex = px-ax
    ey = py-ay

    # seg_len2 == dx**2 + dy**2 == (length of segment)**2, but precalculated
    t = (dx*ex + dy*ey) / seg_len2

    stretch = 100
    capped_t = cap_smooth(t, stretch)

    return (capped_t*dx - ex)**2 + (capped_t*dy - ey)**2

def create_objective(m, dist2_list, init_quality, p):
    if p == -2:
        # Quadratic-harmonic mean
        m.objective = sum(1 / (dist2 + 1e-3) for dist2 in dist2_list)
        m.quality = (len(dist2_list) / m.objective) ** (1/2)
        m.constraints.add(m.objective <= len(dist2_list) / ((init_quality ** 2) + 1e-3))
        m.o = Objective(
            expr=m.objective,
            sense=minimize,
        )
    elif p == -1:
        # Harmonic mean
        m.objective = sum((dist2 + 1e-3) ** (-1/2) for dist2 in dist2_list)
        m.quality = len(dist2_list) / m.objective
        m.constraints.add(m.objective <= len(dist2_list) / (init_quality + 1e-3))
        m.o = Objective(
            expr=m.objective,
            sense=minimize,
        )
    elif p == 0:
        # Geometric mean
        m.objective = sum(log(dist2 + 1e-3) for dist2 in dist2_list)
        m.quality = exp(m.objective / (2*len(dist2_list)))
        m.constraints.add(m.objective >= log(init_quality + 1e-3) * 2 * len(dist2_list))
        m.o = Objective(
            expr=m.objective,
            sense=maximize,
        )
    elif p == 1:
        # Arithmetic mean
        m.objective = sum((dist2 + 1e-3) ** (1/2) for dist2 in dist2_list)
        m.quality = m.objective / len(dist2_list)
        m.constraints.add(m.objective >= init_quality * len(dist2_list))
        m.o = Objective(
            expr=m.objective,
            sense=maximize,
        )
    elif p == 2:
        # Quadratic mean
        m.objective = sum(dist2 for dist2 in dist2_list)
        m.quality = ((m.objective / len(dist2_list)) + 1e-3) ** (1/2)
        m.constraints.add(m.objective >= (init_quality ** 2) * len(dist2_list))
        m.o = Objective(
            expr=m.objective,
            sense=maximize,
        )
    elif p < 0:
        m.objective = sum((dist2 + 1e-3) ** (p/2) for dist2 in dist2_list)
        m.quality = (m.objective / len(dist2_list)) ** (1/p)
        m.constraints.add(m.objective <= ((init_quality + 1e-3) ** p) * len(dist2_list))
        m.o = Objective(
            expr=m.objective,
            sense=minimize,
        )
    elif p > 0:
        m.objective = sum((dist2 + 1e-3) ** (p/2) for dist2 in dist2_list)
        m.quality = (m.objective / len(dist2_list)) ** (1/p)
        m.constraints.add(m.objective >= ((init_quality + 1e-3) ** p) * len(dist2_list))
        m.o = Objective(
            expr=m.objective,
            sense=maximize,
        )
    else:
        raise ValueError('Unsupported mean type: p = {p!r}'.format(**locals()))

def create_pyomo_model(inputdata, init_placement, init_quality):
    area = inputdata.area
    graphs = inputdata_to_graphs(inputdata)
    m = ConcreteModel()
    m.Graph = Set(initialize=graphs.keys())
    m.Point = Set(initialize=[
        (g1, g2, p)
        for (g1, g2), graph in graphs.iteritems()
        for p, point in enumerate(graph.points)
    ])
    m.Segment = Set(initialize=[
        (g1, g2, p, q)
        for (g1, g2), graph in graphs.iteritems()
        for p, q in graph.segments
    ])
    m.c = Var(m.Graph, domain=Reals, bounds=(-1, 1), initialize=lambda m, g1, g2: init_placement.outergroup_placements[g1].innergroup_placements[g2].c)
    m.s = Var(m.Graph, domain=Reals, bounds=(-1, 1), initialize=lambda m, g1, g2: init_placement.outergroup_placements[g1].innergroup_placements[g2].s)
    m.x = Var(m.Graph, domain=Reals, initialize=lambda m, g1, g2: init_placement.outergroup_placements[g1].innergroup_placements[g2].x)
    m.y = Var(m.Graph, domain=Reals, initialize=lambda m, g1, g2: init_placement.outergroup_placements[g1].innergroup_placements[g2].y)
    # (px, py) = transformation of element point
    px_expr = {(g1, g2, p): m.c[g1, g2]*graphs[g1, g2].points[p].x - m.s[g1, g2]*graphs[g1, g2].points[p].y + m.x[g1, g2] for g1, g2, p in m.Point}
    py_expr = {(g1, g2, p): m.s[g1, g2]*graphs[g1, g2].points[p].x + m.c[g1, g2]*graphs[g1, g2].points[p].y + m.y[g1, g2] for g1, g2, p in m.Point}
    m.px = Var(m.Point, domain=Reals, bounds=(0, area.max_x), initialize=lambda m, g1, g2, p: px_expr[g1, g2, p])
    m.py = Var(m.Point, domain=Reals, bounds=(0, area.max_y), initialize=lambda m, g1, g2, p: py_expr[g1, g2, p])
    m.constraints = ConstraintList()
    for g1, g2 in m.Graph:
        # Determinant is 1, i.e. all transformations are orthonormal
        m.constraints.add(m.s[g1, g2]**2 + m.c[g1, g2]**2 == 1)
    for g1, g2, p in m.Point:
        m.constraints.add(m.px[g1, g2, p] == px_expr[g1, g2, p])
        m.constraints.add(m.py[g1, g2, p] == py_expr[g1, g2, p])
    dist2_list = []
    for h1, h2, a, b in m.Segment:
        dx = graphs[h1, h2].points[b].x - graphs[h1, h2].points[a].x
        dy = graphs[h1, h2].points[b].y - graphs[h1, h2].points[a].y
        seg_len2 = dx**2 + dy**2
        for g1, g2, p in m.Point:
            if (h1, h2) != (g1, g2):
                dist2_list.append(
                    dist2_point_segment(
                        m.px[g1, g2, p],
                        m.py[g1, g2, p],
                        m.px[h1, h2, a],
                        m.py[h1, h2, a],
                        m.px[h1, h2, b],
                        m.py[h1, h2, b],
                        seg_len2,
                    )
                )
    create_objective(m, dist2_list, init_quality, p=-1)
    return m

def create_pyomo_solver():
    return SolverFactory('ipopt', options={
        'halt_on_ampl_error': 'yes',
        'max_iter': 3000,
    })

def convert_pyomo_model(inputdata, model):
    m = model
    placement = Placement(
        outergroup_placements=[
            OutergroupPlacement(
                innergroup_placements=[
                    InnergroupPlacement(
                        c=value(m.c[g1, g2]),
                        s=value(m.s[g1, g2]),
                        x=value(m.x[g1, g2]),
                        y=value(m.y[g1, g2]),
                    )
                    for g2, innergroup in enumerate(outergroup.innergroups)
                ]
            )
            for g1, outergroup in enumerate(inputdata.outergroups)
        ],
    )
    return value(m.quality), placement

def get_quality(inputdata, placement):
    model = create_pyomo_model(inputdata, placement, init_quality=0)
    quality, same_placement = convert_pyomo_model(inputdata, model)
    return quality

def run_solver(inputdata, init_placement, init_quality):
    model = create_pyomo_model(inputdata, init_placement, init_quality)
    solver = create_pyomo_solver()
    results = solver.solve(model, keepfiles=False, tee=True)
    if len(results['Solver']) != 1:
        raise ValueError('Number of solvers is not 1')
    if results['Solver'][0]['Status'].key != 'ok':
        raise ValueError('Solver status is not ok')
    quality, placement = convert_pyomo_model(inputdata, model)
    return quality, placement

def optimize(inputdata, random_seed):
    min_improvement_factor = 1.01
    max_retries = 4
    num_retries = 0
    last_placement = None  # Initialized on first loop run
    random_gen = Random()
    random_gen.seed(random_seed)
    init_placement = random_placement(random_gen, inputdata)
    init_quality = get_quality(inputdata, init_placement)
    while True:
        try:
            quality, placement = run_solver(inputdata, init_placement, init_quality)
        except Exception as e:
            if num_retries <= max_retries:
                num_retries += 1
                init_placement = random_placement(random_gen, inputdata)
                continue
            if last_placement is None:
                # First outer iteration failed completely, in all retries, so there is an issue with the model
                raise
            return last_quality, last_placement
        num_retries = 0
        init_quality = quality * min_improvement_factor
        init_placement = placement
        last_quality, last_placement = quality, placement

def run_optimization_loop(inputdata, random_seed):
    quality, placement = optimize(inputdata, random_seed)
    print()
    print('Result: quality={quality!r} placement={placement!r}'.format(**locals()))

def run_single_optimization(inputdata, random_seed):
    random_gen = Random()
    random_gen.seed(0)
    init_placement = random_placement(random_gen, inputdata)
    init_quality = get_quality(inputdata, init_placement)
    quality, placement = run_solver(inputdata, init_placement, init_quality)
    print()
    print('Result: quality={quality!r} placement={placement!r}'.format(**locals()))

def run_model_creation_only(inputdata, random_seed):
    random_gen = Random()
    random_gen.seed(0)
    init_placement = random_placement(random_gen, inputdata)
    init_quality = 20.0
    print('Creating model')
    model = create_pyomo_model(inputdata, init_placement, init_quality)
    print('Writing test_model.nl')
    model.write('test_model.nl')
    print('Done')

if __name__ == '__main__':
    run_model_creation_only(SAMPLE_INPUT_DATA, random_seed=0)
    #run_single_optimization(SAMPLE_INPUT_DATA, random_seed=0)
    #run_optimization_loop(SAMPLE_INPUT_DATA, random_seed=0)
