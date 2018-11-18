#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <functional>
#include <iomanip>
#include <bitset>
#include <queue>
#include <array>
#include <fstream>
#include <cmath>
#include <cassert>
using namespace std;


string Dir[] = { "U", "R", "D", "L" };

class Timer {
	std::chrono::high_resolution_clock::time_point start, end;
	double limit;

public:
	Timer() {
		start = std::chrono::high_resolution_clock::now();
	}
	Timer(double l) {
		start = std::chrono::high_resolution_clock::now();
		limit = l;
	}

	double getTime() {
		end = std::chrono::high_resolution_clock::now();
		return std::chrono::duration<double>(end - start).count();
	}

	bool isOver() {
		if (getTime() > limit) {
			return true;
		}
		return false;
	}

	void setLimit(double l) {
		limit = l;
	}
	void setStart() { start = std::chrono::high_resolution_clock::now(); }
	double getLimit() { return limit; }
};

class Xor128 {
	unsigned static int x, y, z, w;
public:
	static unsigned int rand()
	{
		unsigned int t;
		t = (x ^ (x << 11)); x = y; y = z; z = w;
		return(w = (w ^ (w >> 19)) ^ (t ^ (t >> 8)));
	}

	static void setSeed(unsigned seed)
	{
		x = seed;
	}
};
unsigned int Xor128::x = 31103110, Xor128::y = 123456789, Xor128::z = 521288629, Xor128::w = 88675123;

struct Point
{
	int r, c;
	Point() {}
	Point(int r, int c) :r(r), c(c) {}
};

unsigned RANDs[5][5][20][2];

int COMB[26][26];
void calcCombination()
{
	for (int i = 0; i < 26; i++)
	{
		COMB[i][0] = 1;
		for (int j = 1; j <= i; j++)
		{
			COMB[i][j] = COMB[i - 1][j - 1] * i / j;
		}
	}
}
int tileScore[16];

Point tile_p[25];

int left_table[1048576];
int right_table[1048576];
int merge_table[1048576];
int mono_table[1048576];
int same_table[1048576];
int score_table[1048576];
int smooth_table[1048576];
int empty_table[1048576];
double eval_table[1048576];

int transpose[1048576][5][5];

int reverse_x(int x)
{
	return (x >> 16) | ((x >> 8) & 0x000F0) | (x & 0x00F00) | ((x << 8) & 0x0F000) | ((x << 16) & 0xF0000);
}

int preCalcMonotonicity(int line[])
{
	int mono_left = 0;
	int mono_right = 0;
	int mono_dame = 0;
	for (int j = 2; j < 5; j++)
	{
		if (line[j - 2] == line[j - 1] && line[j - 1] == line[j])
		{

		}
		else if (line[j - 2] >= line[j - 1] && line[j - 1] >= line[j])
		{
			mono_left++;
		}
		else if (line[j - 2] <= line[j - 1] && line[j - 1] <= line[j])
		{
			mono_right++;
		}
		else
		{
			mono_dame++;
		}
	}

	return mono_left + mono_right - mono_dame;
}

int preCalcSmooth(int line[])
{
	int smooth = 0;
	int pre_rank = line[0];
	for (int j = 1; j < 5; j++)
	{
		int rank = line[j];

		smooth += abs(rank - pre_rank);

		pre_rank = rank;
	}

	return smooth;
}

int preCalcSameness(int line[])
{
	int count = 0;
	int sameness = 0;
	int pre_rank = 0;
	for (int j = 0; j < 5; j++)
	{
		int rank = line[j];

		if (rank == 0)continue;

		if (pre_rank == rank)
		{
			count++;
		}
		else if (count > 0)
		{
			sameness += count + 1;
			count = 0;
		}
		pre_rank = rank;
	}
	if (count > 0)sameness += count + 1;

	return sameness;
}

int preCalcEmpty(int line[])
{
	int empty = 0;
	for (int j = 0; j < 5; j++)
	{
		int rank = line[j];

		if (rank == 0)empty++;
	}

	return empty;
}

int preCalcMerge(int line[])
{
	int merge = 0;
	int count = 0;
	int pre_rank = 0;
	for (int j = 0; j < 5; j++)
	{
		int rank = line[j];

		line[j] = 0;

		if (rank == 0)continue;

		if (pre_rank == rank)
		{
			merge++;
			line[count - 1] = rank + 1;
			pre_rank = 0;
		}
		else
		{
			line[count] = rank;

			count++;
			pre_rank = rank;
		}
	}

	return merge;
}

void initTables()
{
	for (int x = 0; x < 1048576; x++)
	{
		int lines[5];
		for (int j = 0; j < 5; j++)lines[j] = (x >> (4 * j)) & 0xf;

		for (int j = 0; j < 5; j++)score_table[x] += tileScore[lines[j]];

		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 5; j++)
			{
				transpose[x][i][j] = lines[j] << (4 * i);
			}
		}

		mono_table[x] = preCalcMonotonicity(lines);

		smooth_table[x] = preCalcSmooth(lines);

		same_table[x] = preCalcSameness(lines);

		empty_table[x] = preCalcEmpty(lines);

		merge_table[x] = preCalcMerge(lines);

		eval_table[x] = 1.5 * empty_table[x] - 1. * smooth_table[x] + 7. * same_table[x] + 1. * mono_table[x];

		int result = 0;
		for (int j = 0; j < 5; j++)result |= lines[j] << (4 * j);
		int rev_result = reverse_x(result);
		int rev_x = reverse_x(x);

		left_table[x] = result;
		right_table[rev_x] = rev_result;
	}
}

void Initializer()
{
	for (int i = 0; i < 5; i++)for (int j = 0; j < 5; j++)for (int k = 0; k < 20; k++)for (int my = 0; my < 2; my++)RANDs[i][j][k][my] = Xor128::rand();
	calcCombination();
	for (int i = 0; i < 5; i++)for (int j = 0; j < 5; j++)tile_p[i * 5 + j] = Point(i, j);

	tileScore[0] = 0;
	tileScore[1] = 0;
	for (int k = 2; k < 16; k++)
	{
		//tileScore[k] = 2 * tileScore[k - 1] + (1 << k);
		tileScore[k] = 2 * tileScore[k - 1] + k;
	}

	initTables();
}


void rows_to_cols(array<int, 5>& rows, array<int, 5>& cols)
{
	for (int i = 0; i < 5; i++)cols[i] = 0;

	for (int i = 0; i < 5; i++)
	{
		int* m = transpose[rows[i]][i];
		for (int j = 0; j < 5; j++)
		{
			cols[j] |= m[j];
		}
	}
}
void cols_to_rows(array<int, 5>& cols, array<int, 5>& rows)
{
	for (int i = 0; i < 5; i++)rows[i] = 0;

	for (int i = 0; i < 5; i++)
	{
		int* m = transpose[cols[i]][i];
		for (int j = 0; j < 5; j++)
		{
			rows[j] |= m[j];
		}
	}
}


struct Board
{
	static const int H, W;

	array<int, 5>rows;
	array<int, 5>cols;

	Board()
	{
		for (int i = 0; i < H; i++)rows[i] = 0;
		for (int j = 0; j < W; j++)cols[j] = 0;
	}

	int getCell(int r, int c)
	{
		return (rows[r] >> (4 * c)) & 0xf;
	}
	void setCell(int r, int c, int rank)
	{
		rows[r] |= rank << (4 * c);
		cols[c] |= rank << (4 * r);
	}


	double evaluation()
	{
		double eval_lr = 0;
		for (int i = 0; i < H; i++)
		{
			eval_lr += eval_table[rows[i]];
		}

		double eval_ud = 0;
		for (int j = 0; j < W; j++)
		{
			eval_ud += eval_table[cols[j]];
		}

		return eval_lr + eval_ud;
	}


	int scoring()
	{
		int score = 0;
		for (int i = 0; i < H; i++)
		{
			score += score_table[rows[i]];
		}
		return score;
	}

	int mergeCount(int dir)
	{
		if (dir % 2)return mergeLR();
		else return mergeUD();
	}

	bitset<25> emptyCells()
	{
		bitset<25>is_empty;
		for (int i = 0; i < H; i++)
		{
			for (int j = 0; j < W; j++)
			{
				int rank = (rows[i] >> (4 * j)) & 0xf;
				if (rank == 0)
				{
					is_empty.set(i*W + j);
				}
			}
		}
		return is_empty;
	}

	unsigned toHash(int my)
	{
		unsigned hash = 0;

		for (int i = 0; i < H; i++)
		{
			for (int j = 0; j < W; j++)
			{
				int rank = (rows[i] >> (4 * j)) & 0xf;
				hash ^= RANDs[i][j][rank][my];
			}
		}

		return hash;
	}

	void reset()
	{
		for (int i = 0; i < H; i++)rows[i] = 0;
		for (int j = 0; j < W; j++)cols[j] = 0;
	}


	int mergeUD()
	{
		int merge = 0;
		for (int j = 0; j < W; j++)
		{
			merge += merge_table[cols[j]];
		}
		return merge;
	}
	int mergeLR()
	{
		int merge = 0;
		for (int i = 0; i < H; i++)
		{
			merge += merge_table[rows[i]];
		}
		return merge;
	}


	int emptyCount()
	{
		int empty_lr = 0;
		for (int i = 0; i < H; i++)
		{
			empty_lr += empty_table[rows[i]];
		}

		int empty_ud = 0;
		for (int j = 0; j < W; j++)
		{
			empty_ud += empty_table[cols[j]];
		}

		return (empty_lr + empty_ud) / 2;
	}

};
const int Board::H = 5, Board::W = 5;



class GameAction
{

	static bool MoveUp(Board& board)
	{
		bool valid = false;
		for (int j = 0; j < Board::W; j++)
		{
			int next = left_table[board.cols[j]];
			if (board.cols[j] != next)valid = true;
			board.cols[j] = next;
		}

		cols_to_rows(board.cols, board.rows);

		return valid;
	}
	static bool MoveDown(Board& board)
	{
		bool valid = false;
		for (int j = 0; j < Board::W; j++)
		{
			int next = right_table[board.cols[j]];
			if (board.cols[j] != next)valid = true;
			board.cols[j] = next;
		}

		cols_to_rows(board.cols, board.rows);

		return valid;
	}
	static bool MoveRight(Board& board)
	{
		bool valid = false;
		for (int i = 0; i < Board::H; i++)
		{
			int next = right_table[board.rows[i]];
			if (board.rows[i] != next)valid = true;
			board.rows[i] = next;
		}

		rows_to_cols(board.rows, board.cols);

		return valid;
	}
	static bool MoveLeft(Board& board)
	{
		bool valid = false;
		for (int i = 0; i < Board::H; i++)
		{
			int next = left_table[board.rows[i]];
			if (board.rows[i] != next)valid = true;
			board.rows[i] = next;
		}

		rows_to_cols(board.rows, board.cols);

		return valid;
	}

public:
	static bool Move(Board& board, int dir)
	{
		if (dir == 0)return MoveUp(board);
		else if (dir == 1)return MoveRight(board);
		else if (dir == 2)return MoveDown(board);
		else if (dir == 3)return MoveLeft(board);

		return false;
	}


	static int AddCell(Board& board, int pos, int rank)
	{
		if (board.getCell(pos / Board::W, pos % Board::W) != 0)return -1;

		board.setCell(pos / Board::W, pos % Board::W, rank);

		return 0;
	}
};

struct ORDER
{
	int dir;
	vector<Point> pos;
};

struct STATE
{
	int player;
	Board board[2];
};

struct DATA
{
	double lower, upper;
	DATA(){}
	DATA(double l, double u):lower(l),upper(u){}
};


bool My_next_permutation(bool array[], int size)
{
	if (size < 2)return false;

	int i = size - 1;
	for (;;)
	{
		int i2 = i;
		i--;

		if (!array[i] && array[i2])
		{
			int j = size;
			while (array[i] || !array[i] && !array[--j]) {}

			swap(array[i], array[j]);
			reverse(array + i2, array + size);

			return true;
		}

		if (i == 0)
		{
			reverse(array, array + size);
			return false;
		}
	}
}


class kurenAI
{
	bool second;
	int turn;
	int timeLeft;
	const int TERM = 1000;

	Board Boards[2];

	int Scores[2];

	Timer tmr;
	bool timeOvered;
	int Depth = 6;
	double timeLimit;

	int loop;
	double pre_eval = 0;
	vector<Point>Ps;
	unordered_map<unsigned, DATA>boardHash;
public:

	void Init()
	{
		cin >> second;

		Boards[0] = Board();
		Boards[1] = Board();
		Boards[0].setCell(2, 2, 1);
		Boards[1].setCell(2, 2, 1);

		cout << "3 3\n";
		cout.flush();
	}

	double alphabeta(STATE&node, ORDER&ret, int depth, double alpha, double beta, int maxDepth)
	{
		Board tmp_myBoard = node.board[node.player];
		Board tmp_enemyBoard = node.board[node.player ^ 1];

		if (depth == maxDepth)
		{
			/*double myeval = tmp_myBoard.evaluation();
			for (int dir = 0; dir < 4; dir++)
			{
				Board myBoard = tmp_myBoard;
				GameAction::Move(myBoard, dir);
				double eval = myBoard.evaluation();
				if (eval > myeval)myeval = eval;
			}
			return myeval - tmp_enemyBoard.evaluation();
			*/
			return tmp_myBoard.evaluation() - tmp_enemyBoard.evaluation();
		}


		unsigned hash = tmp_myBoard.toHash(node.player) ^ tmp_enemyBoard.toHash(node.player ^ 1);
		auto itr = boardHash.find(hash);
		if (itr != boardHash.end())
		{
			DATA data = itr->second;
			if(data.lower >= beta) return data.lower;
			if(data.upper <= alpha) return data.upper;
			if(data.lower > alpha)alpha = data.lower;
			if(data.upper < beta)beta = data.upper;
		}
		else
		{
			boardHash[hash] = DATA(-1e8, 1e8);
		}


		bitset<25>empties = tmp_enemyBoard.emptyCells();
		int Nempty = empties.count();
		int x = empties.to_ulong();


		int kouho[25];
		int size = 0;
		for (int i = 0; i < 25; i++) {
			if (x & (1 << i))
			{
				kouho[size++] = i;
			}
		}


		int merges[4];
		Board myBoards[4];
		int search_counts[4];
		bool search_ends[4];
		bool bitss[4][25];

		pair<double, int>evalSorted[4];

		int pushed = 0;
		for (int dir = 0; dir < 4; dir++)
		{
			search_counts[dir] = 0;
			search_ends[dir] = false;

			myBoards[dir] = tmp_myBoard;


			merges[dir] = myBoards[dir].mergeCount(dir);
			bool valid = GameAction::Move(myBoards[dir], dir);
			if (valid == false)
			{
				search_ends[dir] = true;
				continue;
			}

			evalSorted[pushed] = make_pair(-myBoards[dir].evaluation(), dir);

			pushed++;
		}

		if (pushed == 0)return -1e9;

		double max_value = alpha;
		sort(evalSorted, evalSorted + pushed);
		int index = 0;
		while (pushed)
		{
			loop++;

			if (loop == 1000)
			{
				if (tmr.getTime() > timeLimit)
				{
					loop--;
					timeOvered = true;
					break;
				}
				loop = 0;
			}

			int dir = evalSorted[index].second;

			int n = 1;
			int n_rank = 0;

			if (Nempty <= 4)
			{				
				int x = 0;
				while (1) {
					if (x + COMB[Nempty][n] > search_counts[dir])break;
					x += COMB[Nempty][n];
					n *= 2;
					n_rank++;
					if (n > Nempty || n > merges[dir] + 1)break;
				}

				if (n > Nempty || n > merges[dir] + 1)
				{
					search_ends[dir] = true;
					pushed--;
					index++;
					continue;
				}

				if (x == search_counts[dir])
				{
					for (int i = 0; i < Nempty; i++)bitss[dir][i] = false;
					for (int i = 0; i < n; i++)bitss[dir][Nempty - 1 - i] = true;
				}
			}
			else
			{
				if (search_counts[dir] == size)
				{
					search_ends[dir] = true;
					pushed--;
					index++;
					continue;
				}
			}			

			int rank = merges[dir] + 1 - n_rank;

			Board enemyBoard = tmp_enemyBoard;

			int r = 0;
			if(depth == 0)Ps.clear();
			if(n > 1)
			{
				for (int i = 0; i < n; i++)
				{
					while (bitss[dir][r] == false)r++;
					Point pos(kouho[r] / Board::W, kouho[r] % Board::W);
					if(depth == 0)Ps.push_back(pos);
					enemyBoard.setCell(pos.r, pos.c, rank);
					r++;
				}
				//next_permutation(bitss[k].begin(), bitss[k].end());
				My_next_permutation(bitss[dir], Nempty);
			}
			else
			{
				r = search_counts[dir];
				Point pos = tile_p[kouho[r]];
				if (depth == 0)Ps.push_back(pos);
				enemyBoard.setCell(pos.r, pos.c, rank);
			}
			/*int rank = merges[dir] + 1;


			Board enemyBoard = tmp_enemyBoard;


			int k_ind = search_counts[dir];
			Point pos = tile_p[kouho[k_ind]];
			enemyBoard.setCell(pos.r, pos.c, rank);
			*/

			STATE newNode;
			newNode.board[node.player] = myBoards[dir];
			newNode.board[node.player ^ 1] = enemyBoard;
			newNode.player = node.player ^ 1;

			double next_beta = -max(alpha, max_value);
			double value = -alphabeta(newNode, ret, depth + 1, -beta, next_beta, maxDepth);
			if(value > max_value)
			{
				max_value = value;
				if (depth == 0)
				{
					ret.dir = dir;
					ret.pos = Ps;
				}
			}

			if (max_value >= beta)break;


			search_counts[dir]++;
			/*if (search_counts[dir] == size)
			{
				search_ends[dir] = true;
				pushed--;
				index++;
			}*/
		}

		if(max_value <= alpha)boardHash[hash].upper = max_value;
		else if(max_value >= beta)boardHash[hash].lower = max_value;
		else boardHash[hash] = DATA(max_value, max_value);

		return max_value;
	}

	ORDER minimax()
	{
		tmr.setStart();
		timeOvered = false;
		loop = 0;
		timeLimit = timeLeft * 0.95*1e-5;
		boardHash.clear();
		ORDER ret;
		ret.dir = -1;

		STATE node;
		node.board[0] = Boards[0];
		node.board[1] = Boards[1];
		node.player = 0;

		cerr << alphabeta(node, ret, 0, -1e9, 1e9, Depth) << endl;
		cerr << Depth << endl;
		cerr << loop << endl;
		cerr << tmr.getTime() << endl;

		if (timeOvered) {
			alphabeta(node, ret, 0, -1e9, 1e9, 5);
		}

		if (timeOvered)Depth--;
		else if (tmr.getTime() < timeLimit*1e-2)Depth++;

		return ret;
	}

	ORDER mtdf()
	{
		tmr.setStart();
		timeOvered = false;
		loop = 0;
		timeLimit = timeLeft * 0.95*1e-5;
		boardHash.clear();
		ORDER ret;
		ret.dir = -1;

		STATE node;
		node.board[0] = Boards[0];
		node.board[1] = Boards[1];
		node.player = 0;

		DATA data(-1e9, 1e9);
		double bound = pre_eval;

		while (data.lower < data.upper)
		{
			if(timeOvered)break;
			
			double value = alphabeta(node, ret, 0, bound-1, bound, Depth);

			if(value < bound)data.upper = value;
			else data.lower = value;
			if(data.lower == value)bound = value+1;
			else bound = value;
		}
		pre_eval = bound;

		if(timeOvered){
			loop = -1e9;
			pre_eval = alphabeta(node, ret, 0, -1e9, 1e9, 5);
		}

		cerr << bound << endl;
		cerr << Depth << endl;
		cerr << loop << endl;
		cerr << tmr.getTime() << endl;

		if (timeOvered)Depth--;
		else if (tmr.getTime() < timeLimit*1e-2)Depth++;

		return ret;
	}

	void Input()
	{
		cin >> turn >> timeLeft >> Scores[0] >> Scores[1];

		for (int i = 0; i < Board::H; i++)
		{
			for (int j = 0; j < Board::W; j++)
			{
				int rank; cin >> rank;

				Boards[0].setCell(i, j, rank);
			}
		}

		for (int i = 0; i < Board::H; i++)
		{
			for (int j = 0; j < Board::W; j++)
			{
				int rank; cin >> rank;

				Boards[1].setCell(i, j, rank);
			}
		}
	}


	void Output(int dir, vector<Point> Pos, int rank)
	{
		GameAction::Move(Boards[0], dir);
		for(int i=0;i<Pos.size();i++)GameAction::AddCell(Boards[1], Pos[i].r*Board::W + Pos[i].c, rank);


		string output = Dir[dir];

		output += " " + to_string(Pos.size()) + " " + to_string(rank-(int)log2(Pos.size()));

		for (int i = 0; i < Pos.size(); i++)output += " " + to_string(Pos[i].r + 1) + " " + to_string(Pos[i].c + 1);

		cout << output << endl;
	}

	int MainPhase(ORDER& ord)
	{
		return ord.dir;
	}

	vector<Point> AttackPhase(ORDER& ord)
	{
		return ord.pos;
	}

	int myTurn()
	{
		Boards[0].reset();
		Boards[1].reset();
		Input();

		//ORDER order = minimax();
		ORDER order = mtdf();

		int dir = MainPhase(order);
		if (dir == -1) {
			cout << "ƒ_ƒ" << endl;
			return -1;
		}

		int merge = Boards[0].mergeCount(dir);

		vector<Point> pos = AttackPhase(order);
		Output(dir, pos, merge + 1);

		return 0;
	}

	void start()
	{
		Init();

		for (int t = 0; t < TERM; t++)
		{
			myTurn();
		}
	}
};


int main() {
	Initializer();

	kurenAI kurenai3110;
	kurenai3110.start();

	return 0;
}