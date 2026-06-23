// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_externalT.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "memory.h"
#include "update.h"

#include <cstring>
#include <iostream>

using namespace LAMMPS_NS;
using namespace FixConst;

enum{PF_CALLBACK,PF_ARRAY};

/* ---------------------------------------------------------------------- */

FixExternalT::FixExternalT(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  fexternalT(nullptr), caller_vector(nullptr)
{
  if (narg < 4) error->all(FLERR,"Illegal fix externalT command");

  scalar_flag = 1;
  global_freq = 1;
  extscalar = 1;
  energy_global_flag = energy_peratom_flag = 1;
  virial_global_flag = virial_peratom_flag = 1;
  thermo_energy = thermo_virial = 1;

  if (strcmp(arg[3],"pf/callback") == 0) {
    if (narg != 6) error->all(FLERR,"Illegal fix externalT command");
    mode = PF_CALLBACK;
    ncall = utils::inumeric(FLERR,arg[4],false,lmp);
    napply = utils::inumeric(FLERR,arg[5],false,lmp);
    if (ncall <= 0 || napply <= 0)
      error->all(FLERR,"Illegal fix externalT command");
  } else if (strcmp(arg[3],"pf/array") == 0) {
    if (narg != 5) error->all(FLERR,"Illegal fix externalT command");
    mode = PF_ARRAY;
    napply = utils::inumeric(FLERR,arg[4],false,lmp);
    if (napply <= 0) error->all(FLERR,"Illegal fix externalT command");
  } else error->all(FLERR,"Illegal fix externalT command");

  callback = nullptr;

  // perform initial allocation of atom-based array
  // register with Atom class

  FixExternalT::grow_arrays(atom->nmax);
  atom->add_callback(Atom::GROW);

  user_energy = 0.0;
  memset(user_virial, 0, sizeof(user_virial));

  // optional vector of values provided by caller
  // vector_flag and size_vector are setup via set_vector_length()

  caller_vector = nullptr;
}

/* ---------------------------------------------------------------------- */

FixExternalT::~FixExternalT()
{
  // unregister callbacks to this fix from Atom class

  atom->delete_callback(id,Atom::GROW);

  memory->destroy(fexternalT);
  delete[] caller_vector;
}

/* ---------------------------------------------------------------------- */

int FixExternalT::setmask()
{
  int mask = 0;
  if (mode == PF_CALLBACK || mode == PF_ARRAY) {
    mask |= PRE_REVERSE;
    mask |= POST_FORCE;
    mask |= MIN_POST_FORCE;
  }
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixExternalT::init()
{
  if (mode == PF_CALLBACK && callback == nullptr)
    error->all(FLERR,"Fix externalT callback function not set");
}

/* ---------------------------------------------------------------------- */

void FixExternalT::setup(int vflag)
{
  post_force(vflag);
}

/* --------------------------------------------------------------------- */

void FixExternalT::setup_pre_reverse(int eflag, int vflag)
{
  pre_reverse(eflag,vflag);
}

/* ---------------------------------------------------------------------- */

void FixExternalT::min_setup(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   store eflag, so can use it in post_force to tally per-atom energies
------------------------------------------------------------------------- */

void FixExternalT::pre_reverse(int eflag, int /*vflag*/)
{
  eflag_caller = eflag;
}

/* ---------------------------------------------------------------------- */

void FixExternalT::post_force(int vflag)
{
  bigint ntimestep = update->ntimestep;

  int eflag = eflag_caller;
  ev_init(eflag,vflag);

  // invoke the callback in driver program
  // it will fill fexternalT with forces

  if (mode == PF_CALLBACK && ntimestep % ncall == 0)
    (this->callback)(ptr_caller,update->ntimestep,
                     atom->nlocal,atom->tag,atom->x,fexternalT);

  // add forces from fexternalT to atoms in group

  if (ntimestep % napply == 0) {
    double **torque = atom->torque;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
	      //std::cout<<i<<"\t"<<fexternalT[i][0]<<"\t"<<fexternal[i][1]<<"\t"<<fexternal[i][2]<<"\n";
        torque[i][0] += fexternalT[i][0];
        torque[i][1] += fexternalT[i][1];
        torque[i][2] += fexternalT[i][2];
      }

    // add contribution to global virial from previously stored value

    if (vflag_global)
      for (int i = 0; i < 6; ++i)
        virial[i] = user_virial[i];
  }
}

/* ---------------------------------------------------------------------- */

void FixExternalT::min_post_force(int vflag)
{
  post_force(vflag);
}

// ----------------------------------------------------------------------
// "set" methods caller can invoke directly
// ----------------------------------------------------------------------

/* ----------------------------------------------------------------------
   caller invokes this method to set its contribution to global energy
   this is the *total* energy across all MPI ranks of the externalT code
   and must be set for all MPI ranks.
   unlike other energy/virial set methods:
     do not just return if eflag_global is not set
     b/c input script could access this quantity via compute_scalar()
     even if eflag is not set on a particular timestep
   this function is compatible with CALLBACK and ARRAY mode
------------------------------------------------------------------------- */

void FixExternalT::set_energy_global(double caller_energy)
{
  user_energy = caller_energy;
}

/* ----------------------------------------------------------------------
   caller invokes this method to set its contribution to the global virial
   for all MPI ranks.  the virial value is the *total* contribution across
   all MPI ranks of the externalT code and thus we need to divide by the
   number of MPI ranks since the tallying code expects per MPI rank contributions.
   this function is compatible with PF_CALLBACK and PF_ARRAY mode
------------------------------------------------------------------------- */

void FixExternalT::set_virial_global(double *caller_virial)
{
  const double npscale = 1.0/(double)comm->nprocs;
  for (int i = 0; i < 6; i++)
    user_virial[i] = npscale * caller_virial[i];
}

/* ----------------------------------------------------------------------
   caller invokes this method to set its contribution to peratom energy.
   this is applied to the *local* atoms only.
   this function is compatible with PF_CALLBACK mode only since it tallies
   its energy contributions directly into the accumulator arrays.
------------------------------------------------------------------------- */

void FixExternalT::set_energy_peratom(double *caller_energy)
{
  if (!eflag_atom) return;
  if ((mode == PF_ARRAY) && (comm->me == 0))
    error->warning(FLERR,"Can only set energy/atom for fix externalT in pf/callback mode");

  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++)
    eatom[i] = caller_energy[i];
}

/* ----------------------------------------------------------------------
   caller invokes this method to set its contribution to peratom virial
   this is applied to the *local* atoms only.
   this function is compatible with PF_CALLBACK mode only since it tallies
   its virial contributions directly into the accumulator arrays.
------------------------------------------------------------------------- */

void FixExternalT::set_virial_peratom(double **caller_virial)
{
  int i,j;

  if (!vflag_atom) return;
  if ((mode == PF_ARRAY) && (comm->me == 0))
    error->warning(FLERR,"Can only set virial/atom for fix externalT in pf/callback mode");

  int nlocal = atom->nlocal;
  for (i = 0; i < nlocal; i++)
    for (j = 0; j < 6; j++)
      vatom[i][j] = caller_virial[i][j];
}

/* ----------------------------------------------------------------------
   caller invokes this method to set length of global vector of values
   assume all vector values are extensive.
------------------------------------------------------------------------- */

void FixExternalT::set_vector_length(int n)
{
  delete[] caller_vector;

  vector_flag = 1;
  size_vector = n;
  extvector = 1;

  caller_vector = new double[n];
}

/* ----------------------------------------------------------------------
   caller invokes this method to set value for item at "index" in vector
   index is 1-based, thus index ranges from 1 to N inclusively.
   Must be called from all MPI ranks.
------------------------------------------------------------------------- */

void FixExternalT::set_vector(int index, double value)
{
  if (index > size_vector)
    error->all(FLERR,"Invalid set_vector index ({} of {}) in fix externalT",index,size_vector);
  caller_vector[index-1] = value;
}

/* ----------------------------------------------------------------------
   potential energy of added force
   up to user to set it via set_energy()
------------------------------------------------------------------------- */

double FixExternalT::compute_scalar()
{
  return user_energy;
}

/* ----------------------------------------------------------------------
   arbitrary value computed by caller
   up to user to set it via set_vector()
------------------------------------------------------------------------- */

double FixExternalT::compute_vector(int n)
{
  return caller_vector[n];
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixExternalT::memory_usage()
{
  double bytes = 3*atom->nmax * sizeof(double);
  bytes += 6*sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate atom-based array
------------------------------------------------------------------------- */

void FixExternalT::grow_arrays(int nmax)
{
  memory->grow(fexternalT,nmax,3,"externalT:fexternalT");
  memset(&fexternalT[0][0], 0, sizeof(double)*3*nmax);
}

/* ----------------------------------------------------------------------
   copy values within local atom-based array
------------------------------------------------------------------------- */

void FixExternalT::copy_arrays(int i, int j, int /*delflag*/)
{
  fexternalT[j][0] = fexternalT[i][0];
  fexternalT[j][1] = fexternalT[i][1];
  fexternalT[j][2] = fexternalT[i][2];
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

int FixExternalT::pack_exchange(int i, double *buf)
{
  buf[0] = fexternalT[i][0];
  buf[1] = fexternalT[i][1];
  buf[2] = fexternalT[i][2];
  return 3;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
------------------------------------------------------------------------- */

int FixExternalT::unpack_exchange(int nlocal, double *buf)
{
  fexternalT[nlocal][0] = buf[0];
  fexternalT[nlocal][1] = buf[1];
  fexternalT[nlocal][2] = buf[2];
  return 3;
}

/* ----------------------------------------------------------------------
   externalT caller sets a callback function to invoke in post_force()
------------------------------------------------------------------------- */

void FixExternalT::set_callback(FnPtr caller_callback, void *caller_ptr)
{
  callback = caller_callback;
  ptr_caller = caller_ptr;
}

/* ----------------------------------------------------------------------
   get access to internal data structures
------------------------------------------------------------------------- */

void *FixExternalT::extract(const char *str, int &dim)
{
  if (strcmp(str, "fexternalT") == 0) {
    dim = 2;
    return (void *) fexternalT;
  }
  return nullptr;
}
