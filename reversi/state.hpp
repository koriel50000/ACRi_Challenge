#ifdef REVERSI_STATE_HPP_
#define REVERSI_STATE_HPP_

class State {
public:
	void convertBuffer(float[ROWS * COLUMNS * CHANNELS]);
private:
	uint64_t player;
	uint64_t opponent;
}

#endif // REVERSI_STATE_HPP_
