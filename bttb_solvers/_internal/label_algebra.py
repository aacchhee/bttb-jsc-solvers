# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
from itertools import product
from bttb_solvers._internal.halfstrip_core.exact import PrimeField, pd, pm, pgcd, ps
from bttb_solvers._internal.native_field_view import NativeBaseFieldView
from bttb_solvers._internal.owned_label_algebra import OwnedLabelAlgebra


def prime_label_modulus(p, degree):
    """algorithm finite lexicographic search, Frobenius/gcd criterion.

    Retains counts and only the accepted certificate, not every rejected
    candidate. PrimeField and all degree-small polynomial arithmetic are the
    existing exact implementation. No caller-field metadata chooses p.
    """
    if type(p) is not int or p < 2 or type(degree) is not int or degree < 1:
        raise ValueError('positive degree and discovered prime characteristic required')
    F = PrimeField(p)
    counts = {'candidates': 0, 'gcd_tests': 0,
              'binary_power_calls': 0,
              'modular_polynomial_products': 0,
              'rejected_by_gcd': 0, 'rejected_final_frobenius': 0}
    def remainder(value, modulus):
        return pd(F, value, modulus)[1]
    def power_mod(value, exponent, modulus):
        counts['binary_power_calls'] += 1
        result = [F.one()]
        while exponent:
            if exponent & 1:
                result = remainder(pm(F, result, value), modulus)
                counts['modular_polynomial_products'] += 1
            exponent //= 2
            if exponent:
                value = remainder(pm(F, value, value), modulus)
                counts['modular_polynomial_products'] += 1
        return result
    for coefficients in product(range(p), repeat=degree):
        counts['candidates'] += 1
        modulus = list(coefficients) + [1]
        x = remainder([0, 1], modulus)
        frobenius = x
        gcd_certificate = []
        rejected = False
        for i in range(1, degree+1):
            frobenius = power_mod(frobenius, p, modulus)
            if i <= degree//2:
                gcd = pgcd(F, ps(F, frobenius, x), modulus)
                counts['gcd_tests'] += 1
                if len(gcd) != 1:
                    counts['rejected_by_gcd'] += 1
                    rejected = True
                    break
                gcd_certificate.append({'frobenius_index': i, 'gcd': gcd})
        if rejected:
            continue
        if ps(F, frobenius, x):
            counts['rejected_final_frobenius'] += 1
            continue
        return modulus, {'criterion': 'gcd(X^(p^i)-X,f)=1 for 1<=i<=floor(e/2); X^(p^e)=X mod f',
                         'accepted_gcds': gcd_certificate,
                         'final_frobenius_residue': frobenius,
                         'x_residue': x, 'counts': dict(counts)}
    raise ArithmeticError('finite prime-field irreducible enumeration exhausted')


class LabelAlgebraContext:
    """One attempt's field, flat embedded coefficients and label catalog.

    Native elements are never converted through integers. Only an algebra
    added by this context is removed on physical-output descent.
    """
    def __init__(self, native_base, field, coefficients, labels, metadata, added):
        self.native_base = native_base
        self.field = field
        self.coefficients = tuple(coefficients)
        self.labels = tuple(labels)
        self.metadata = metadata
        self.added_algebra = added

    def embed(self, value):
        return self.field.embed(value) if self.added_algebra else value

    def descend(self, value):
        if not self.added_algebra:
            return value

        
        constant = value[0]
        if not self.field.is_zero(self.field.sub(value, self.field.embed(constant))):
            raise ArithmeticError('physical output did not descend to caller base')
        return constant


def setup_label_algebra(base, coefficients, M, restriction=None):
    """Generic M-label service; preserves the half-strip M=n contract."""
    return _setup_label_algebra(base, coefficients, M, restriction)


def _setup_label_algebra(base, coefficients, M, restriction=None, modulus_policy=None):
    """Embed a FLAT native coefficient iterable, constructing M unit labels.

    Probe actual integer images through M. A first zero discovers p<=M;
    otherwise integer images 1..M suffice without knowing characteristic.
    Small-characteristic labels are nonzero base-p digit polynomials of the
    least degree e with p**e>=M+1, reduced in the added retained algebra.
    Restriction must be a monic proper factor of that initial polynomial in
    the caller's base. The global owner supplies it after a Restart.

    Ingress performs O(number of supplied coefficients) algebra embeddings;
    each added-algebra element has degree-dependent storage/scalar overhead.
    No dense Toeplitz blocks or coefficient-matrix tensor is built.
    """
    if isinstance(M, bool) or not isinstance(M, int) or M < 0:
        raise ValueError('M must be a nonnegative integer')
    native = base
    base = NativeBaseFieldView(base)
    probes = []

    
    integer_images = [base.zero()]
    one = base.one()
    characteristic = None
    for integer in range(1, M+1):
        image = base.add(integer_images[-1], one)
        integer_images.append(image)
        zero = base.is_zero(image)
        probes.append({'integer': integer, 'image': image, 'zero': zero})
        if zero:
            characteristic = integer
            break
    if characteristic == 1:
        raise ValueError('effective field has zero identity')
    construction = None
    if characteristic is None:
        if restriction is not None:
            raise ValueError('restriction supplied without an added algebra')
        field = base
        labels = integer_images[1:]
        embed = lambda value: value
        degree = 1
        added = False
    else:
        p = characteristic
        degree = 1
        cardinality = p
        while cardinality < M+1:
            degree += 1
            cardinality *= p
        if modulus_policy is None:
            polynomial, certificate = prime_label_modulus(p, degree)
        else:
            degree, polynomial, certificate = modulus_policy(p, M, degree)

        
        prime_images = integer_images[:p]
        initial = [prime_images[value] for value in polynomial]
        modulus = initial
        if restriction is not None:
            modulus = list(restriction)
            if not 2 <= len(modulus) < len(initial):
                raise ValueError('restriction must have positive smaller degree')
            if not base.is_zero(base.sub(modulus[-1], base.one())):
                raise ValueError('restriction must be monic')
            _quotient, remainder = pd(base, initial, modulus)
            if any(not base.is_zero(value) for value in remainder):
                raise ValueError('restriction is not a factor of initial algebra')
        field = OwnedLabelAlgebra(base, modulus)
        labels = []
        for code in range(1, M+1):
            value = code
            digits = []
            for _ in range(degree):
                digits.append(prime_images[value % p])
                value //= p
            labels.append(field.reduce(digits))
        embed = field.embed
        added = True
        construction = {'characteristic_from_probes': p,
                        'initial_extension_degree': degree,
                        'prime_modulus': polynomial,
                        'irreducibility_certificate': certificate,
                        'candidates_visited': certificate['counts']['candidates'],
                        'implementation': 'prime_label_modulus: Frobenius/gcd criterion over existing exact.PrimeField polynomial routines',
                        'theorem_cost_scope': 'O(M^2 polylog(M+1)) setup from algorithm / algorithm; covered by concrete tau for half-strip M=n; no automatic banded M=2kn budget claim'}
    embedded = []
    for value in coefficients:
        embedded.append(embed(value))
    metadata = {'required_labels': M, 'characteristic_probes': probes,
                'integer_image_additions': len(probes),
                'integer_embedding': 'repeated addition of one, cached native image map; caller from_int unused',
                'characteristic_detected': characteristic,
                'added_algebra': added, 'initial_extension_degree': degree,
                'retained_degree': field.degree if added else None,
                'restriction': None if restriction is None else tuple(modulus),
                'construction': construction, 'coefficient_count': len(embedded),
                'coefficient_ingress': 'one pass over flat native coefficients; no dense block expansion',
                'restart_owner': 'global dispatcher; discard all attempt state on Restart',
                'label_cost': 'M probes at most; M*initial_degree digit/reduction work when extended; inherited irreducible search charged separately'}
    return LabelAlgebraContext(native, field, embedded, labels, metadata, added)


def quadratic_label_modulus(p):
    """algorithm A: first Euler-certified nonsquare, after p is discovered."""
    if type(p) is not int or p <= 2 or p % 2 == 0:
        raise ValueError('discovered odd prime characteristic required')
    F = PrimeField(p)
    exponent = (p-1)//2
    minus_one = F.neg(F.one())
    products = 0
    tests = 0
    for candidate in range(1, p):
        tests += 1
        value = F.from_int(candidate)
        result = F.one()
        remaining = exponent
        while remaining:
            if remaining & 1:
                result = F.mul(result, value)
                products += 1
            remaining //= 2
            if remaining:
                value = F.mul(value, value)
                products += 1
        if F.is_zero(F.sub(result, minus_one)):
            polynomial = [F.neg(F.from_int(candidate)), F.zero(), F.one()]
            return polynomial, {'criterion': 'Euler nonsquare in discovered prime field',
                                'selected_a': candidate, 'exponent': exponent,
                                'power_residue': result, 'minus_one': minus_one,
                                'counts': {'candidates': tests, 'binary_power_calls': tests,
                                           'prime_field_products': products, 'gcd_tests': 0,
                                           'modular_polynomial_products': 0}}
    raise ArithmeticError('Euler nonsquare scan exhausted odd prime field')


def setup_banded_label_algebra(base, coefficients, n, k, restriction=None):
    """Dimension-aware algorithm A catalog of M=2kn+1 nonzero labels.

    This helper is for the narrow caller's paid setup; the wide raw-generator
    route does not need it. It constructs one guarded attempt, not dispatch.
    """
    if type(n) is not int or n < 2 or type(k) is not int or not 1 <= k < n:
        raise ValueError('integer n>=2 and 1<=k<n required')
    D, M = 2*k*n, 2*k*n+1
    selected = {}
    def policy(p, required, minimum_degree):
        if p <= 2*n:
            polynomial, certificate = prime_label_modulus(p, minimum_degree)
            selected.update(branch='small_prime_frobenius', implementation='lex monic Frobenius/gcd criterion')
            return minimum_degree, polynomial, certificate
        if p*p < required+1:
            raise ArithmeticError('algorithm quadratic catalog cardinality invariant failed')
        polynomial, certificate = quadratic_label_modulus(p)
        selected.update(branch='large_prime_quadratic', implementation='deterministic integer-order Euler nonsquare scan')
        return 2, polynomial, certificate
    context = _setup_label_algebra(base, coefficients, M, restriction, modulus_policy=policy)
    metadata = context.metadata
    branch = selected.get('branch', 'native_integer_labels')
    context.label_ids = tuple(range(M))
    metadata.update(n=n, k=k, D=D, M=M, branch=branch,
                    stable_original_label_ids=context.label_ids,
                    catalog_creation_count=M,
                    catalog_digit_positions=M*metadata['initial_extension_degree'] if context.added_algebra else 0,
                    coefficient_ingress_count=metadata['coefficient_count'],
                    cost_scope='algorithm A: O(k n^2 polylog(n+2)) setup; not a full narrow solver or wide-route setup requirement')
    if context.added_algebra:
        construction = metadata['construction']
        p = metadata['characteristic_detected']

        
        prime_images = [context.field.base.zero()] + [probe['image'] for probe in metadata['characteristic_probes'][:p-1]]
        construction.update(implementation=selected['implementation'],
                            theorem_cost_scope=metadata['cost_scope'],
                            initial_native_modulus=tuple(prime_images[value] for value in construction['prime_modulus']),
                            retained_native_modulus=tuple(context.field.modulus))
        metadata['construction_counts'] = dict(construction['irreducibility_certificate']['counts'])
    else:
        metadata['construction_counts'] = {'candidates': 0, 'binary_power_calls': 0,
                                           'prime_field_products': 0, 'gcd_tests': 0,
                                           'modular_polynomial_products': 0}
    return context
