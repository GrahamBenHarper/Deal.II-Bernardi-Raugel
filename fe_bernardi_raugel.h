// ---------------------------------------------------------------------
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
// 
//
// ---------------------------------------------------------------------

#ifndef dealii_fe_bernardi_raugel_h
#define dealii_fe_bernardi_raugel_h

#include <deal.II/base/config.h>
#include <deal.II/base/table.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/tensor_product_polynomials.h>
#include <deal.II/base/polynomials_bernardi_raugel.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_poly_tensor.h>

#include <vector>

DEAL_II_NAMESPACE_OPEN

/**
 * The Bernardi-Raugel element.
 *
 * <h3>Degrees of freedom</h3>
 * The BR1 element has <i>dim</i> degrees of freedom on each node and 1 on each
 * face. 
 */
template <int dim>
class FE_BernardiRaugel
  :
  public FE_PolyTensor<PolynomialsBernardiRaugel<dim>, dim>
{
public:
  /**
   * Constructor for the Bernardi-Raugel element of degree @p p.
   * Currently only supports <i>p=1</i>.
   */
  FE_BernardiRaugel (const unsigned int p = 1);

  /**
   * Return a string that uniquely identifies a finite element. This class
   * returns <tt>FE_BR<dim>(degree)</tt>, with @p dim and @p degree replaced
   * by appropriate values.
   */
  virtual std::string get_name () const;

  virtual
  std::unique_ptr<FiniteElement<dim,dim> >
  clone() const;

  // documentation inherited from the base class
  virtual
  void
  convert_generalized_support_point_values_to_dof_values (const std::vector<Vector<double> > &support_point_values,
                                                          std::vector<double>                &nodal_values) const;

private:
  /**
   * Only for internal use. Its full name is @p get_dofs_per_object_vector
   * function and it creates the @p dofs_per_object vector that is needed
   * within the constructor to be passed to the constructor of @p
   * FiniteElementData.
   */
  static std::vector<unsigned int>
  get_dpo_vector (const unsigned int degree);

  /**
   * Initialize the FiniteElement<dim>::generalized_support_points and
   * FiniteElement<dim>::generalized_face_support_points fields. Called from
   * the constructor. See the
   * @ref GlossGeneralizedSupport "glossary entry on generalized support points"
   * for more information.
   */
  void initialize_support_points (const unsigned int bdm_degree);
};

DEAL_II_NAMESPACE_CLOSE

#endif
