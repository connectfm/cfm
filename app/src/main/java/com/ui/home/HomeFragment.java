package com.ui.home;

import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.FrameLayout;

import androidx.annotation.NonNull;
import androidx.coordinatorlayout.widget.CoordinatorLayout;
import androidx.fragment.app.Fragment;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.cfm.recommend.Recommender;
import com.example.cfm.R;
import java.util.ArrayList;
import java.util.Set;
import java.util.stream.Collectors;

import com.google.android.material.snackbar.Snackbar;
import com.spotifyFramework.Song;
import com.spotifyFramework.SongService;

public class HomeFragment extends Fragment {
	private SharedPreferences.Editor editor;
	private SharedPreferences preferences;
	public SongService songService;
	private RecyclerView recyclerView;
	private SongsAdapter adapter;
	private Recommender recommender;
	private Button radioStart;
	private ArrayList<String> recommendations;
	private int recommendationSize = 3;
	private int iteration = 1;
	private Snackbar snackbar;

	public View onCreateView(@NonNull LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
		preferences = getActivity().getSharedPreferences("SPOTIFY", 0);
		View root = inflater.inflate(R.layout.fragment_home, container, false);
		radioStart = (Button) root.findViewById(R.id.begin_listening);
		recommendations = new ArrayList<>();
		recommender = new Recommender(root.getContext());
		songService = new SongService(getActivity());
		songService.getRecentlyPlayed(() -> {
			ArrayList<Song> recents = songService.getPlaylist();
			recents = removeDupes(recents);
			adapter = new SongsAdapter(getActivity(), recents);
			adapter.notifyDataSetChanged();
			recyclerView = (RecyclerView) getActivity().findViewById(R.id.listening_history);
			recyclerView.setAdapter(adapter);
			recyclerView.setLayoutManager(new LinearLayoutManager(getActivity()));
		});
		getRecommendations();
			snackbar = Snackbar.make(root,R.string.please_wait, Snackbar.LENGTH_SHORT);
			View view = snackbar.getView();
			CoordinatorLayout.LayoutParams params = (CoordinatorLayout.LayoutParams)view.getLayoutParams();
			params.gravity = Gravity.TOP;
			view.setLayoutParams(params);
			radioStart.setOnClickListener(new View.OnClickListener() {
				@Override
				public void onClick(View v) {
					if(recommendations.size() < 3) {
						System.out.println(recommendations.size());
						snackbar.show();
					}
					else {
						NavController navController = Navigation
								.findNavController(getActivity(), R.id.nav_host_fragment);
						navController.navigate(R.id.song_dashboard);
					}
				}
			});

		return root;
	}

	private ArrayList<Song> removeDupes(ArrayList<Song> playlist) {
		ArrayList<String> idList = new ArrayList<>();
		ArrayList<Song> res = new ArrayList<>();

		for(Song s: playlist) {
			if(!idList.contains(s.getUri())){
				res.add(s);
				idList.add(s.getUri());
			}
		}
		return res;
	}
	private void getRecommendations() {
		for (int i = 1; i <= recommendationSize; i++) {
			recommender.get("01", () -> {
				recommendations.add(recommender.getRecommenation());
				editor = preferences.edit();
				editor.putString("id_" + iteration, recommender.getRecommenation());
				editor.apply();
				iteration++;

				System.out.println("first:: " + preferences.getString("id_1", "nothing"));
				System.out.println("second:: " + preferences.getString("id_2", "nothing"));
				System.out.println("third:: " + preferences.getString("id_3", "nothing"));

			});
		}
	}
}