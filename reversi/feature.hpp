#ifdef REVERSI_FEATURE_HPP_
#define REVERSI_FEATURE_HPP_

class Feature {
public:
	Feature();
	void clear();
	void setState(const uint64_t player, const uint64_t opponent);
	void getStateBuffer(const uint64_t player, const uint64_t opponent, float[ROWS * COLUMNS * CHANNELS]);
private:
	State current_state;

}

#endif // REVERSI_FEATURE_HPP_
