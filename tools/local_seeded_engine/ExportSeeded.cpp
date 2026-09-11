// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and
// character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
//
// Local competition-testing adapter. This translation unit intentionally
// includes the unmodified official exporter and adds one opt-in entry point.
// It must not be packaged with a competition submission.

#include <cstdint>

#include "../../dataset/ptcg_engine/ptcgProgram 22/Export.cpp"

namespace {

StartData ApiBattleStartSeeded(int* cards, std::uint32_t seed) {
  ApiData* data = new ApiData();
  data->apiDataType = 1;

  GameConfig config = {};
  // Avoid Game::init's seed==0 random_device fallback; the exact requested
  // seed is restored immediately after initialization below.
  config.seed = (seed == 0 ? 1U : seed);
  config.recordLog = true;
  // Every in-battle random branch must use Game::rng. Merely seeding rng while
  // leaving this true would still use std::random_device for shuffles, coins,
  // and random target ordering.
  config.deviceRand = false;

  // Keep the official BattleStart deck validation byte-for-byte equivalent in
  // meaning. This is duplicated here so the official source remains untouched.
  for (int i = 0; i < 2; i++) {
    std::unordered_map<std::u8string, int> nameCount;
    bool aceSpec = false;
    bool basic = false;
    for (int j = 0; j < DECK_SIZE; j++) {
      CardId id = cards[i * DECK_SIZE + j];
      if (!CardTable.contains(id)) {
        delete data;
        return {nullptr, i, 1};
      }

      const CardMaster& master = CardTable.at(id);
      if (master.aceSpec) {
        if (aceSpec) {
          delete data;
          return {nullptr, i, 4};
        }
        aceSpec = true;
      }

      if (master.cardType == CardType::Pokemon &&
          master.evolutionType == EvolutionType::Basic) {
        basic = true;
      }

      int& count = nameCount[master.name];
      count++;
      if (count > DECK_SAME_CARD_MAX &&
          master.cardType != CardType::BasicEnergy) {
        delete data;
        return {nullptr, i, 2};
      }

      config.decks[i].cards[j] = cards[i * DECK_SIZE + j];
    }
    if (!basic) {
      delete data;
      return {nullptr, i, 3};
    }
  }

  data->init(config);

  // Restore the requested value so every uint32 seed, including zero, is
  // represented exactly in both the config and generator.
  data->game.config.seed = seed;
  data->game.config.deviceRand = false;
  data->game.rng = std::mt19937(seed);

  data->start();
  data->next();
  return {data, -1, 0};
}

}  // namespace

extern "C" {

GAME_API StartData BattleStartSeeded(int* cards, std::uint32_t seed) {
  return ApiBattleStartSeeded(cards, seed);
}

}  // extern "C"
