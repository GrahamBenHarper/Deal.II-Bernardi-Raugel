/*
 * A gutted version of the step-20 code written to work with
 * Bernardi-Raugel elements. The linear system is solved directly,
 * which reduces the need for most of the original code.
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>

#include <deal.II/fe/fe_bernardi_raugel.h>
#include <deal.II/base/tensor_function.h>


namespace Step20
{
  using namespace dealii;
  
  template <int dim>
  class MixedLaplaceProblem
  {
  public:
    MixedLaplaceProblem (const unsigned int degree);
    void run ();

  private:
    void make_grid_and_dofs ();
    void assemble_system ();
    void solve ();
    void compute_errors () const;
    void output_results () const;

    const unsigned int   degree;

    Triangulation<dim>   triangulation;
    FESystem<dim>        fe;
    DoFHandler<dim>      dof_handler;

    BlockSparsityPattern      sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;
    ConstraintMatrix          constraints;

    BlockVector<double>       solution;
    BlockVector<double>       system_rhs;
  };

  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide () : Function<dim>(1) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };



  template <int dim>
  class PressureBoundaryValues : public Function<dim>
  {
  public:
    PressureBoundaryValues () : Function<dim>(1) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };


  template <int dim>
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution () : Function<dim>(dim+1) {}

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
  };


  template <int dim>
  double RightHandSide<dim>::value (const Point<dim>  &p,
                                    const unsigned int /*component*/) const
  {
    //2d sinsin example
    return 2*3.14159*3.14159*std::sin(3.14159*p[0])*std::sin(3.14159*p[1]);

    //3d sinsinsin example
    //return 3*3.14159*3.14159*std::sin(3.14159*p[0])*std::sin(3.14159*p[1])*std::sin(3.13159*p[2]);

    //other 2d examples
    //return 0;
  }



  template <int dim>
  double PressureBoundaryValues<dim>::value (const Point<dim>  &p,
                                             const unsigned int /*component*/) const
  {
    const double alpha = 0.3;
    const double beta = 1;

    //original example
    //return -(alpha*p[0]*p[1]*p[1]/2 + beta*p[0] - alpha*p[0]*p[0]*p[0]/6);

    //2d and 3d sinsinsin examples
    return 0;

    //p(x,y)=-x example
    //return -p[0];
  }



  template <int dim>
  void
  ExactSolution<dim>::vector_value (const Point<dim> &p,
                                    Vector<double>   &values) const
  {
    Assert (values.size() == dim+1,
            ExcDimensionMismatch (values.size(), dim+1));

    const double alpha = 0.3;
    const double beta = 1;

    //2d original example
    //values(0) = alpha*p[1]*p[1]/2 + beta - alpha*p[0]*p[0]/2;
    //values(1) = alpha*p[0]*p[1];
    //values(2) = -(alpha*p[0]*p[1]*p[1]/2 + beta*p[0] - alpha*p[0]*p[0]*p[0]/6);
    
    //2d sinsin example
    values(0) = -3.14159*std::cos(3.14159*p[0])*std::sin(3.14159*p[1]);
    values(1) = -3.14159*std::sin(3.14159*p[0])*std::cos(3.14159*p[1]);
    values(2) = std::sin(3.14159*p[0])*std::sin(3.14159*p[1]);

    //3d sinsinsin example
    //values(0) = -3.14159*std::cos(3.14159*p[0])*std::sin(3.14159*p[1])*std::sin(3.14159*p[2]);
    //values(1) = -3.14159*std::sin(3.14159*p[0])*std::cos(3.14159*p[1])*std::sin(3.14159*p[2]);
    //values(2) = -3.14159*std::sin(3.14159*p[0])*std::sin(3.14159*p[1])*std::cos(3.14159*p[2]);
    //values(3) = std::sin(3.14159*p[0])*std::sin(3.14159*p[1])*std::sin(3.14159*p[2]);

    //2d p(x,y)=-x example
    //values(0) = 1;
    //values(1) = 0;
    //values(2) = -p[0];
  }


  /*
   * A class to handle the permeability tensor K. In all examples, K is taken
   * to be the identity, so K inverse is also the identity.
   */
  template <int dim>
  class KInverse : public TensorFunction<2,dim>
  {
  public:
    KInverse () : TensorFunction<2,dim>() {}

    virtual void value_list (const std::vector<Point<dim> > &points,
                             std::vector<Tensor<2,dim> >    &values) const;
  };


  template <int dim>
  void
  KInverse<dim>::value_list (const std::vector<Point<dim> > &points,
                             std::vector<Tensor<2,dim> >    &values) const
  {
    Assert (points.size() == values.size(),
            ExcDimensionMismatch (points.size(), values.size()));

    for (unsigned int p=0; p<points.size(); ++p)
      {
        values[p].clear ();

        for (unsigned int d=0; d<dim; ++d)
          values[p][d][d] = 1.;
      }
  }

  /*
   * The class for the mixed laplacian problem. The pair of spaces
   * for this problem are BR1 with DGQ0.
   */
  template <int dim>
  MixedLaplaceProblem<dim>::MixedLaplaceProblem (const unsigned int degree)
    :
    degree (degree),
    fe (FE_BernardiRaugel<dim>(degree+1), 1,
        FE_DGQ<dim>(degree), 1),
    dof_handler (triangulation)
  {}

  /*
   * Set up the degrees of freedom and domain for the problem. All
   * test examples are on the cube [-1,1]^3 with varying levels
   * of refinement.
   */
  template <int dim>
  void MixedLaplaceProblem<dim>::make_grid_and_dofs ()
  {
    GridGenerator::hyper_cube (triangulation, -1, 1);
    triangulation.refine_global (7);

    dof_handler.distribute_dofs (fe);

    DoFRenumbering::component_wise (dof_handler);
    constraints.close();

    std::vector<types::global_dof_index> dofs_per_component (dim+1);
    DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);
    const unsigned int n_u = dofs_per_component[0],
                       n_p = dofs_per_component[dim];

    std::cout << "Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl
              << "Total number of cells: "
              << triangulation.n_cells()
              << std::endl
              << "Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << " (" << n_u << '+' << n_p << ')'
              << std::endl;

    BlockDynamicSparsityPattern dsp(2, 2);
    dsp.block(0, 0).reinit (n_u, n_u);
    dsp.block(1, 0).reinit (n_p, n_u);
    dsp.block(0, 1).reinit (n_u, n_p);
    dsp.block(1, 1).reinit (n_p, n_p);
    dsp.collect_sizes ();
    DoFTools::make_sparsity_pattern (dof_handler, dsp);

    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit (sparsity_pattern);

    solution.reinit (2);
    solution.block(0).reinit (n_u);
    solution.block(1).reinit (n_p);
    solution.collect_sizes ();

    system_rhs.reinit (2);
    system_rhs.block(0).reinit (n_u);
    system_rhs.block(1).reinit (n_p);
    system_rhs.collect_sizes ();
  }

  /*
   * Assemble the global system, leaving the boundary conditions
   * enforced naturally.
   */
  template <int dim>
  void MixedLaplaceProblem<dim>::assemble_system ()
  {
    QGauss<dim>   quadrature_formula(3);
    QGauss<dim-1> face_quadrature_formula(3);

    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values    | update_gradients |
                             update_quadrature_points  | update_JxW_values);
    FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                      update_values    | update_normal_vectors |
                                      update_quadrature_points  | update_JxW_values);

    const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int   n_q_points      = quadrature_formula.size();
    const unsigned int   n_face_q_points = face_quadrature_formula.size();

    FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       local_rhs (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    const RightHandSide<dim>          right_hand_side;
    const PressureBoundaryValues<dim> pressure_boundary_values;
    const KInverse<dim>               k_inverse;

    std::vector<double> rhs_values (n_q_points);
    std::vector<double> boundary_values (n_face_q_points);
    std::vector<Tensor<2,dim> > k_inverse_values (n_q_points);

    const FEValuesExtractors::Vector velocities (0);
    const FEValuesExtractors::Scalar pressure (dim);

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      {
        fe_values.reinit (cell);
        local_matrix = 0;
        local_rhs = 0;

        right_hand_side.value_list (fe_values.get_quadrature_points(),
                                    rhs_values);
        k_inverse.value_list (fe_values.get_quadrature_points(),
                              k_inverse_values);

        for (unsigned int q=0; q<n_q_points; ++q)
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              const Tensor<1,dim> phi_i_u     = fe_values[velocities].value (i, q);
              const double        div_phi_i_u = fe_values[velocities].divergence (i, q);
              const double        phi_i_p     = fe_values[pressure].value (i, q);

              for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                  const Tensor<1,dim> phi_j_u     = fe_values[velocities].value (j, q);
                  const double        div_phi_j_u = fe_values[velocities].divergence (j, q);
                  const double        phi_j_p     = fe_values[pressure].value (j, q);

                  local_matrix(i,j) += (phi_i_u * k_inverse_values[q] * phi_j_u
                                        - div_phi_i_u * phi_j_p
                                        - phi_i_p * div_phi_j_u)
                                       * fe_values.JxW(q);
                }

              local_rhs(i) += -phi_i_p *
                              rhs_values[q] *
                              fe_values.JxW(q);
            }

        for (unsigned int face_n=0;
             face_n<GeometryInfo<dim>::faces_per_cell;
             ++face_n)
          if (cell->at_boundary(face_n))
            {
              fe_face_values.reinit (cell, face_n);

              pressure_boundary_values
              .value_list (fe_face_values.get_quadrature_points(),
                           boundary_values);

              for (unsigned int q=0; q<n_face_q_points; ++q)
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                  local_rhs(i) += -(fe_face_values[velocities].value (i, q) *
                                    fe_face_values.normal_vector(q) *
                                    boundary_values[q] *
                                    fe_face_values.JxW(q));
            }


        cell->get_dof_indices (local_dof_indices);

      constraints.distribute_local_to_global (local_matrix,
                                              local_rhs,
                                              local_dof_indices,
                                              system_matrix,
                                              system_rhs);
      }
  }

  /*
   * Solve the linear system directly and time it.
   */
  template <int dim>
  void MixedLaplaceProblem<dim>::solve ()
  {
    std::cout << "Solving linear system... ";
    Timer timer;

    SparseDirectUMFPACK  A_direct;
    A_direct.initialize(system_matrix);

    A_direct.vmult (solution, system_rhs);
    constraints.distribute (solution);

    timer.stop ();
    std::cout << "done ("
            << timer.cpu_time()
            << "s)"
            << std::endl;
  }

  /*
   * Compute the errors for this problem. Use a degree 4 quadrature,
   * although anywhere in the 2-8 degree range produces similar results.
   */
  template <int dim>
  void MixedLaplaceProblem<dim>::compute_errors () const
  {
    const ComponentSelectFunction<dim>
    pressure_mask (dim, dim+1);
    const ComponentSelectFunction<dim>
    velocity_mask(std::make_pair(0, dim), dim+1);

    ExactSolution<dim> exact_solution;
    Vector<double> cellwise_errors (triangulation.n_active_cells());

    QTrapez<1>     q_trapez;
    QIterated<dim> quadrature (q_trapez, 4);

    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature,
                                       VectorTools::L2_norm,
                                       &pressure_mask);
    const double p_l2_error = VectorTools::compute_global_error(triangulation,
                                                                cellwise_errors,
                                                                VectorTools::L2_norm);

    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature,
                                       VectorTools::L2_norm,
                                       &velocity_mask);
    const double u_l2_error = VectorTools::compute_global_error(triangulation,
                                                                cellwise_errors,
                                                                VectorTools::L2_norm);

    std::cout << "Errors: ||e_p||_L2 = " << p_l2_error
              << ",   ||e_u||_L2 = " << u_l2_error
              << std::endl;
  }

  /*
   * Output the data to a VTU file for visualization
   */

  template <int dim>
  void MixedLaplaceProblem<dim>::output_results () const
  {
    std::vector<std::string> solution_names(dim, "u");
    solution_names.push_back ("p");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation (dim,
                    DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    data_out.add_data_vector (dof_handler, solution, solution_names, interpretation);

    data_out.build_patches (degree);

    std::ofstream output ("solution.vtu");
    data_out.write_vtu (output);
  }

  /*
   * Run the system
   */
  template <int dim>
  void MixedLaplaceProblem<dim>::run ()
  {
    make_grid_and_dofs();
    assemble_system ();
    solve ();
    compute_errors ();
    output_results ();
  }
}

/*
 * The main piece of code. Solve a 2D or 3D mixed laplace equation
 */
int main ()
{
  try
    {
      using namespace dealii;
      using namespace Step20;

      MixedLaplaceProblem<2> mixed_laplace_problem(0);
//      MixedLaplaceProblem<3> mixed_laplace_problem(0);
      mixed_laplace_problem.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
