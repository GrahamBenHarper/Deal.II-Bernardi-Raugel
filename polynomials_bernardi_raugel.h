// ---------------------------------------------------------------------
//
// HEADER SPACE FOR LATER USE 
//
// 
//
// 
// 
// 
// 
// 
// 
//
// ---------------------------------------------------------------------


#ifndef dealii_polynomials_bernardi_raugel_h
#define dealii_polynomials_bernardi_raugel_h

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/polynomial_space.h>
#include <deal.II/base/tensor_product_polynomials.h>
#include <deal.II/base/geometry_info.h>
#include <vector>

DEAL_II_NAMESPACE_OPEN


/**
 * This class implements the Bernardi-Raugel polynomials of degree 1 as
 * described in their <i>Mathematics of Computation</i> paper from 1985.
 *
 * The Bernardi-Raugel polynomials are originally defined as an enrichment
 * of the <i>Q<sub>1</sub><sup>d</sup></i> elements for Stokes problems
 * by the addition of bubble functions, yielding a locking-free finite
 * element method.
 *
 * There are dim shape functions per vertex and 1 shape function per face.
 * On one element, the shape functions are ordered by the <i>Q<sub>1</sub><sup>d</sup></i>
 * dim shape functions supported on each vertex, increasing in order of the
 * vertex ordering on the element, then the remaining bubble functions are
 * defined in order of the faces.
 * 
 * On a given face, the corresponding bubble function is defined to be zero
 * on all other faces and on all vertices. When restricted to the face, it
 * should be quadratic in dim-1 variables, reaching a maximum value of 1 on
 * the center of the face.
 *
 * @ingroup Polynomials
 * @author Graham Harper
 * @date 2018
 */
template <int dim>
class PolynomialsBernardiRaugel
{
public:
  /**
   * Constructor. Creates all basis functions for Bernardi-Raugel polynomials
   * of given degree.
   *
   * @arg k: the degree of the Bernardi-Raugel-space, which is currently
   * limited to the case <tt>k=1</tt>
   */
  PolynomialsBernardiRaugel(const unsigned int k);

  /**
   * Return the number of Bernardi-Raugel polynomials.
   */
  unsigned int n () const;


  /**
   * Return the degree of Bernardi-Raugel polynomials.
   * Since the bubble functions are quadratic in at least one variable,
   * the degree of the Bernardi-Raugel polynomials is two.
   */
  unsigned int degree () const;

  /**
   * Return the name of the space, which is <tt>BernardiRaugel</tt>.
   */
  std::string name () const;

  /**
   * Compute the value and the first and second derivatives of each Bernardi-
   * Raugel polynomial at @p unit_point.
   *
   * The size of the vectors must either be zero or equal <tt>n()</tt>.  In
   * the first case, the function will not compute these values.
   *
   * If you need values or derivatives of all tensor product polynomials then
   * use this function, rather than using any of the <tt>compute_value</tt>,
   * <tt>compute_grad</tt> or <tt>compute_grad_grad</tt> functions, see below,
   * in a loop over all tensor product polynomials.
   */
  void compute(const Point<dim> &unit_point,
               std::vector<Tensor<1,dim> > &values,
               std::vector<Tensor<2,dim> > &grads,
               std::vector<Tensor<3,dim> > &grad_grads,
               std::vector<Tensor<4,dim> > &third_derivatives,
               std::vector<Tensor<5,dim> > &fourth_derivatives) const;
  /**
   * Return the number of polynomials in the space <tt>BR(degree)</tt> without
   * requiring to build an object of PolynomialsBernardiRaugel. This is
   * required by the FiniteElement classes.
   */
  static unsigned int compute_n_pols(unsigned int degree);


private:

  /**
   * The degree of this object given to the constructor (must be 1).
   */
  const unsigned int my_degree;

  /**
   * The number of Bernardi-Raugel polynomials.
   */
  const unsigned int n_pols;

  /**
   * An object representing the polynomial space of Q
   * functions which forms the <tt>BR</tt> polynomials through
   * outer products of these with the corresponding unit ijk
   * vectors.
   */  
  const AnisotropicPolynomials<dim> polynomial_space_Q;
  /**
   * An object representing the polynomial space of bubble
   * functions which forms the <tt>BR</tt> polynomials through
   * outer products of these with the corresponding normals.
   */
  const AnisotropicPolynomials<dim> polynomial_space_bubble;

  /**
   * A static member function that creates the polynomial space we use to
   * initialize the #polynomial_space_Q member variable.
   */
  static
  std::vector<std::vector< Polynomials::Polynomial< double > > >
  create_polynomials_Q (const unsigned int k);

  /**
   * A static member function that creates the polynomial space we use to
   * initialize the #polynomial_space_bubble member variable.
   */
  static
  std::vector<std::vector< Polynomials::Polynomial< double > > >
  create_polynomials_bubble (const unsigned int k);
};


template <int dim>
inline
unsigned int
PolynomialsBernardiRaugel<dim>::n() const
{
  return n_pols;
}

template <int dim>
inline
unsigned int
PolynomialsBernardiRaugel<dim>::degree() const
{
  return 2;
}

template <int dim>
inline
std::string
PolynomialsBernardiRaugel<dim>::name() const
{
  return "BernardiRaugel";
}


DEAL_II_NAMESPACE_CLOSE

#endif
