// Copyright (c) 2009-2011, Tor M. Aamodt, Inderpreet Singh
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "scoreboard.h"
#include "../cuda-sim/ptx_sim.h"
#include "shader.h"
#include "shader_trace.h"
#include "../../libcuda/gpgpu_context.h"

// SCOREBOARD_DPRINTF disabled default
//#define SCOREBOARD_DPRINTF printf
#define SCOREBOARD_DPRINTF

// Constructor
Scoreboard::Scoreboard(unsigned sid, unsigned n_warps, class gpgpu_t* gpu)
    : longopregs() {
  m_sid = sid;
  // Initialize size of table
  reg_table.resize(n_warps);
  longopregs.resize(n_warps);

  m_gpu = gpu;
}

// Print scoreboard contents
void Scoreboard::printContents() const {
  bool is_empty = true;
  SCOREBOARD_DPRINTF("scoreboard contents (sid=%d): \n", m_sid);
  for (unsigned i = 0; i < reg_table.size(); i++) {
    if (reg_table[i].size() == 0) continue;
    SCOREBOARD_DPRINTF("  wid = %2d: ", i);
    for (auto it = reg_table[i].begin(); it != reg_table[i].end(); it++) {
      SCOREBOARD_DPRINTF("%u(", it->first);
      print_active_mask(it->second);
      SCOREBOARD_DPRINTF(") ");
    }
    SCOREBOARD_DPRINTF("\n");
    is_empty = false;
  }
  if (is_empty) {
    SCOREBOARD_DPRINTF("[N/A]\n");
  }
}

void Scoreboard::reserveRegister(unsigned wid, active_mask_t msk, unsigned regnum) {
  SCOREBOARD_DPRINTF("[ZSY][Scoreboard] reserveRegister() wid = %d, active_mask = ", wid);
  print_active_mask(msk);
  SCOREBOARD_DPRINTF(", regnum = %d\n", regnum);
  auto iter = reg_table[wid].find(regnum);
  active_mask_t sb_msk;
  sb_msk.reset();
  if (iter != reg_table[wid].end()) {
    sb_msk = iter->second;
    if ((sb_msk & msk).any()) {
      printf(
        "Error: trying to reserve an already reserved register (sid=%d, "
        "wid=%d, regnum=%d).",
        m_sid, wid, regnum);
      abort();
    }
  }
  sb_msk |= msk;
  reg_table[wid][regnum] = sb_msk;
}

// Unmark register as write-pending
void Scoreboard::releaseRegister(unsigned wid, active_mask_t msk, unsigned regnum) {
  SCOREBOARD_DPRINTF("[ZSY][Scoreboard] releaseRegister() wid = %d, active_mask = ", wid);
  print_active_mask(msk);
  SCOREBOARD_DPRINTF(", regnum = %d at cycle %d\n", regnum, m_gpu->gpgpu_ctx->clock());
  auto iter = reg_table[wid].find(regnum);
  assert(iter != reg_table[wid].end());
  auto& sb_msk = iter->second;
  auto tmp_msk= sb_msk;
  tmp_msk &= msk;
  tmp_msk ^= msk;
  assert(!tmp_msk.any()); // the bit to be cleared must be set!
  sb_msk &= ~msk;
  if (!sb_msk.any()) {
    reg_table[wid].erase(iter);
  }
}

// ZSY Warning! Maybe longopregs need to be modified from std::set to std::map too,
// but it seems only work for 2-level warp scheduler, so I just leave it alone.
const bool Scoreboard::islongop(unsigned warp_id, unsigned regnum) {
  return longopregs[warp_id].find(regnum) != longopregs[warp_id].end();
}

void Scoreboard::reserveRegisters(const class warp_inst_t* inst) {
  for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
    if (inst->out[r] > 0) {
      reserveRegister(inst->original_wid(), inst->get_warp_active_mask(), inst->out[r]);
    }
  }

  // Keep track of long operations
  if (inst->is_load() && (inst->space.get_type() == global_space ||
                          inst->space.get_type() == local_space ||
                          inst->space.get_type() == param_space_kernel ||
                          inst->space.get_type() == param_space_local ||
                          inst->space.get_type() == param_space_unclassified ||
                          inst->space.get_type() == tex_space)) {
    for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
      if (inst->out[r] > 0) {
        //SHADER_DPRINTF(SCOREBOARD, "New longopreg marked - warp:%d, reg: %d\n",
        //               inst->warp_id(), inst->out[r]);
        longopregs[inst->warp_id()].insert(inst->out[r]);
      }
    }
  }
}

// Release registers for an instruction
void Scoreboard::releaseRegisters(const class warp_inst_t* inst) {
  for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
    if (inst->out[r] > 0) {
      //SHADER_DPRINTF(SCOREBOARD, "Register Released - warp:%d, reg: %d\n",
      //               inst->warp_id(), inst->out[r]);
      releaseRegister(inst->original_wid(), inst->get_warp_active_mask(), inst->out[r]);
      longopregs[inst->warp_id()].erase(inst->out[r]);
    }
  }
}

/**
 * Checks to see if registers used by an instruction are reserved in the
 *scoreboard
 *
 * @return
 * true if WAW or RAW hazard (no WAR since in-order issue)
 **/
bool Scoreboard::checkCollision(unsigned wid, active_mask_t msk, const warp_inst_t *inst) const {
  SCOREBOARD_DPRINTF("[ZSY][Scoreboard] checkCollision() wid = %d, active_mask = ", wid);
  print_active_mask(msk);
  SCOREBOARD_DPRINTF(", regnum = ");
  // Get list of all input and output registers
  std::set<int> inst_regs;

  for (unsigned iii = 0; iii < inst->outcount; iii++)
    inst_regs.insert(inst->out[iii]);

  for (unsigned jjj = 0; jjj < inst->incount; jjj++)
    inst_regs.insert(inst->in[jjj]);

  if (inst->pred > 0) inst_regs.insert(inst->pred);
  if (inst->ar1 > 0) inst_regs.insert(inst->ar1);
  if (inst->ar2 > 0) inst_regs.insert(inst->ar2);

  // Check for collision, get the intersection of reserved registers and
  // instruction registers
  std::set<int>::const_iterator it2;
  for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++) {
    SCOREBOARD_DPRINTF("%d ", *it2);
    auto iter = reg_table[wid].find(*it2);
    if (iter != reg_table[wid].end()) {
#if 1 // if we will enable scoreboard optimization, enable default
      auto sb_msk = iter->second;
      sb_msk &= msk;
      if (sb_msk.any()) {
        return true;
      }
#else
        return true;
#endif
    }
  }
  SCOREBOARD_DPRINTF("\n");
  return false;
}

bool Scoreboard::pendingWrites(unsigned wid) const {
  return !reg_table[wid].empty();
}

bool Scoreboard::print_active_mask(active_mask_t msk) const {
  for (int i = 31; i >= 0; i--) {
    SCOREBOARD_DPRINTF("%c", msk.test(i) ? '1' : '0');
  }
}
