package ui.home;

import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.cfm.recommend.Recommender;
import com.example.cfm.R;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import spotify_framework.Song;
import spotify_framework.SongService;

public class HomeFragment extends Fragment {

	public SongService songService;
	private RecyclerView recyclerView;
	private SongsAdapter adapter;
	private Recommender recommender;
	private Button radioStart;
	private SharedPreferences preferences;


	public View onCreateView(@NonNull LayoutInflater inflater,
			ViewGroup container, Bundle savedInstanceState) {
		preferences = getActivity().getSharedPreferences("SPOTIFY", 0);
		View root = inflater.inflate(R.layout.fragment_home, container, false);
		radioStart = (Button) root.findViewById(R.id.begin_listening);
		songService = new SongService(getActivity());
		songService.getRecentlyPlayed(() -> {
			ArrayList<Song> recents = songService.getPlaylist();
			adapter = new SongsAdapter(getActivity(), recents);
			adapter.notifyDataSetChanged();
			recyclerView = (RecyclerView) getActivity().findViewById(R.id.listening_history);
			recyclerView.setAdapter(adapter);
			recyclerView.setLayoutManager(new LinearLayoutManager(getActivity()));
		});
		radioStart.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				//Ask the recommender for a set of songs to place in queue
				//Send most recent location and recent songs w/ neutral rating to DataStore
                /*
                This is for testing to make sure song information is being transferred to the
                playback fragment and that the playback fragment plays the correct songs
                 */
				songService = new SongService(getActivity());
				songService.getRecentlyPlayed(() -> {
					List<Song> songList = songService.getPlaylist();
					Set<String> songSet = new HashSet<String>();

					for (int i = 0; i < songList.size(); i++) {
						System.out.println(songList.get(i).toString());
						songSet.add(songList.get(i).toString());
					}
					SharedPreferences.Editor editor = preferences.edit();
					editor.putStringSet("songs", songSet);
					editor.apply();

					NavController navController = Navigation
							.findNavController(getActivity(), R.id.nav_host_fragment);
					navController.navigate(R.id.song_dashboard);
				});

			}
		});

		return root;
	}

	@Override
	public void onActivityCreated(Bundle savedInstanceState) {
		super.onActivityCreated(savedInstanceState);

	}
}